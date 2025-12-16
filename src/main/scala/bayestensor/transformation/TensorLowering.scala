package bayestensor
package transformation

import ir._
import utils._
import scala.collection.mutable
import scala.collection.mutable.LinkedHashSet

class TensorLowering(val linear: Boolean) extends (Exp => TensorExp) {
  import BayesTensorNormalChecker._
  def apply(e: Exp): TensorExp = {
    // println(s"e: ${e}")
    val pq = fullQuery(e)(FullQueryCtx.empty)
    // println(s"pq: ${pq}")
    val qs = pq.queries.zip(pq.syms)
    // println(s"qs: ${qs.mkString("{{", "\n,", "}}")}")
    // val vars = varOrder(e)
    val cpts = computeCPTs(e)(Map())
    var res: TensorExp = null
    // keeps the mapping between the BTL symbol and its associated tensor symbol
    // var tensorList = Map[Sym, TensorSym]()
    var tensorList = cpts.map(x => 
      x._2 match {
        case TensorConst(_, dim) =>
          x._1 -> TensorSym(x._1.asInstanceOf[Sym].name, dim)
        case TensorLoad(name, args, dim) => 
          x._1 -> TensorSym(name, dim)
        case TensorSym(_, _) =>
          x._1 -> x._2
      }
    ).toMap.asInstanceOf[Map[Sym, TensorSym]]
    // println(s"tensorList: ${tensorList.getClass}")
    // val reverseMap = mutable.Map[String, TensorSym]()
    var einsumSymMap = Map[Sym, Map[Sym, Char]]() //
    var einsumOutputMap = Map[Sym, String]() 
    var einsumSymStart = 0
    var einsumOutput = Map[Sym, Seq[Sym]]() // name of variable in generated file and corresponding einsum output vars
    var unitMatriceTrack = Seq[Sym]()
    for((query, sym) <- qs) {
      // println(s"query: ${query}")
      // println(s"sym: ${sym}")
      query match {
        case query: FullQuery => 
          // println(s"query.dag: ${query.dag}")
          // println(s"query.weights: ${query.weights}")
          // println(s"query: ${query}")
          val vars = varOrder(query)
          // println(s"vars: ${vars}")
          // println(s"tensorList: ${tensorList}")
          val (einsumMap, einsumOutputVars, nres, tensorListUpdate) = if(linear)
            linearVE(query, vars, cpts, einsumSymStart)
          else
            variableElimination(query, vars, cpts, einsumSymStart, tensorList)
          einsumOutputMap += (sym -> einsumOutputVars)
          einsumOutput += (sym -> query.outputVars)
          einsumSymMap += (sym -> einsumMap)
          einsumSymStart += vars.size
          // println(s"tensorListUpdate: $tensorListUpdate")
          val TensorLetN(bindings2, result2) = nres
          // println(s"result2: ${result2}")
          val res2sym = TensorSym(sym.name, result2.dim)
          val bindings1 = 
            if(res == null) 
              Seq() 
            else {
              val TensorLetN(bindings1, result1) = res
              bindings1
            }
          res = TensorLetN(bindings1 ++ bindings2 ++ Seq(res2sym -> result2), res2sym)    
          // println(s"res: ${Seq(res2sym -> result2)}\n")
          // update tensorList due to selections or observations
          for (x <- tensorListUpdate) {
            tensorList.-(x._1)
            tensorList += (x._1 -> x._2)
          }
          tensorList += (sym -> res2sym) // keep tracking the tensor from each computed query
          // println(s"\ntensorList: $tensorList\n\n")
        case exp: ExpQuery => // ExpQuery(normVars, dist, binsMean)
          val TensorLetN(bindings1, result1) = res
          var finalDim = TensorDim.Scalar
          var input1 = new LinkedHashSet[Char]() 
          einsumOutputMap(exp.dist).foreach(x => input1.add(x))
          var input2 = new LinkedHashSet[Char]() 
          var outArgs = new LinkedHashSet[Char]()
          val dist_ten = tensorList(exp.dist)
          val binsMean_ten = tensorList(exp.binsMean)
          val args = Seq(dist_ten, binsMean_ten)
          // println(s"exp.dims: ${exp.dims}")
          if (!exp.dims.isEmpty){
            if (unitMatriceTrack.contains(exp.binsMean)) {
              input2 = input1
            } else {
              val input1Seq = input1.toSeq
              exp.dims.map(i => input2.add(input1Seq(i.asInstanceOf[IntLiteral].i)))
              // input2 = input1 &~ outArgs // FIXME: wrong
            }
          } else {
            throw new Exception("User needs to guide how to compute the expectation by declaring dimensions to act on.")
          }
        
          if (!exp.normVars.isEmpty) {
            val dist_symbols = einsumSymMap(exp.dist)
            // println(s"${sym.name} dist_symbols: $dist_symbols")
            exp.normVars.foreach(nv => outArgs.add(dist_symbols(nv)))
            val dimAmap = (input1 zip dist_ten.dim.dims).toMap
            // println(s"${sym.name} dimAmap: $dimAmap")
            val dims = (for (x <- outArgs.toSeq; if dimAmap.contains(x)) yield dimAmap(x))
            // println(s"${sym.name} dims: $dims")
            finalDim = TensorDim(outArgs.size, dims)
          } // else no output vars

          val einsum = s"${input1.mkString("")},${input2.mkString("")}->${outArgs.mkString("")}"
          val res1sym = TensorSym(sym.name, finalDim)
          res = TensorLetN(bindings1 ++ Seq(res1sym -> TensorContract(args, einsum, finalDim)), res1sym)
          tensorList += (sym -> res1sym)
          einsumOutputMap += (sym -> outArgs.mkString(""))
          einsumSymMap += (Sym(res1sym.name) -> (exp.normVars zip outArgs).toMap)
          einsumOutput += (sym -> exp.normVars)
          // println(s"${sym.name} -> exp.normVars: ${exp.normVars}")
        case DecompressQuery(e) =>
          val TensorLetN(bindings1, result1) = res
          e match {
            case Decompress(e1, e2, args) =>
              val e1Sym = e1.asInstanceOf[Sym]
              val idx_mat = e2.asInstanceOf[Sym]
              if (tensorList.contains(e1Sym) && tensorList.contains(idx_mat)) {
                // einsumSymMap = Map[Sym, Map[Sym, Char]]() //
                // einsumOutputMap = Map[Sym, String]() 
                // einsumSymStart = 0
                // einsumOutput
                if (args.size != 1) {
                  raise("Decompression for more than 1 dimension hasn't been supported yet.")
                } else {
                  val e1TenSym = tensorList(e1Sym)
                  val tenMat = tensorList(idx_mat)
                  // println(s"einsumOutputMap: ${einsumOutputMap}")
                  // println(s"einsumOutput: ${einsumOutput}")
                  var toSumOverDim = Seq[Int]()
                  einsumOutputMap(e1Sym).zipWithIndex.map(
                    x => args.map(a => einsumOutput(e1Sym).indexOf(a)).map(a2 =>
                      if (x._2 == a2) {
                        toSumOverDim = toSumOverDim :+ x._2
                      }
                    )
                  )
                  // println(s"e1sym: ${e1Sym}")
                  // println(s"e1TenSym: ${e1TenSym}")
                  var toSumOver = einsumOutputMap(e1Sym).zipWithIndex.filter(eo => toSumOverDim.contains(eo._2)).map(_._1).mkString("")
                  // println(s"toSumOver: ${toSumOver}")
                  var tenMatUniq = Seq[Char]()
                  for (i <- (0 until tenMat.dim.order-1)) {
                    tenMatUniq = tenMatUniq :+ (97+einsumSymStart).toChar
                    einsumSymStart += 1
                  } 
                  val tenMatSymbol = toSumOver + tenMatUniq.mkString("")
                  var outArgs = ""
                  var dims = Seq[Int]()
                  // println(s"einsumOutputMap(e1Sym): ${einsumOutputMap(e1Sym)}")
                  einsumOutputMap(e1Sym).zipWithIndex.map(es =>
                    if (toSumOverDim.contains(es._2)) {
                      outArgs += tenMatUniq.mkString("")
                      dims ++= tenMat.dim.dims.drop(1)
                    } else {
                      outArgs += es._1
                      dims = dims :+ e1TenSym.dim.dims(es._2)
                    }
                  )
                  val einsum = s"${einsumOutputMap(e1Sym)},${tenMatSymbol}->${outArgs}"
                  val finalDim = TensorDim(dims.size, dims)
                  val res1sym = TensorSym(sym.name, finalDim)
                  res = TensorLetN(bindings1 ++ Seq(res1sym -> TensorContract(Seq(e1TenSym,tenMat), einsum, finalDim)), res1sym)
                  // println(s"TensorContract(Seq(e1TenSym,tenMat), einsum, finalDim): ${TensorContract(Seq(e1TenSym,tenMat), einsum, finalDim)}")
                  assert(finalDim.order == e1TenSym.dim.order)
                  einsumOutput += (sym -> einsumOutput(e1Sym))
                  einsumOutputMap += (sym -> outArgs)
                  tensorList += (sym -> res1sym)
                  einsumSymMap += (sym -> (einsumOutput(e1Sym) zip outArgs).toMap)
                }
              } else {
                raise(s"Cannot decompress because matrice `$e` needed are not loaded into the environment.")
              }
            case _ =>
              raise(s"`$e` is not supported yet.")
          }
        case OuterProductQuery(sym1, sym2) =>
          val TensorLetN(bindings1, result1) = res
          val tensor1 = tensorList(sym1)
          val tensor1Dim = tensor1.dim
          val tensor2 = tensorList(sym2)
          val tensor2Dim = tensor2.dim
          val finalDim = TensorDim(tensor1Dim.order + tensor2Dim.order, tensor1Dim.dims ++ tensor2Dim.dims)
          val res1sym = TensorSym(sym.name, finalDim)
          var einsumSym1 = ""
          var einsumSym2 = ""
          if (einsumOutputMap.contains(sym1)) {
            einsumSym1 = einsumOutputMap(sym1)
          } else {
            einsumSym1 = tensor1Dim.dims.zipWithIndex.map(x => ('a' + einsumSymStart +  x._2).toChar).mkString("")
            einsumSymStart += tensor1Dim.dims.size
          }
          
          // println(s"einsumOutputMap: ${einsumOutputMap}")
          // println(s"sym2: ${sym2}")
          // println(s"")
          if (einsumOutputMap.contains(sym2)) {
            einsumSym2 = einsumOutputMap(sym2)
            // println(s"einsumSym2 2: ${einsumSym2}")
          } else {
            // einsumSym2 = tensor2Dim.dims.zipWithIndex.map(x => (einsumSym1.charAt(einsumSym1.length-1) + einsumSymStart + x._2).toChar).mkString("")
            einsumSym2 = tensor2Dim.dims.zipWithIndex.map(x => ('a' + einsumSymStart + x._2).toChar).mkString("")
            // println(s"einsumSym2 3: ${einsumSym2}")
            einsumSymStart += tensor2Dim.dims.size
          }
          val einsum = s"${einsumSym1},${einsumSym2}->${einsumSym1+einsumSym2}"
          res = TensorLetN(bindings1 ++ Seq(res1sym -> TensorContract(Seq(tensor1, tensor2), einsum, finalDim)), res1sym)
          tensorList += (Sym(res1sym.name) -> res1sym)
          einsumOutputMap += (Sym(res1sym.name) -> s"${einsumSym1+einsumSym2}")
          val outputSym1 = if (einsumOutput.contains(sym1)) {einsumOutput(sym1)} else {Seq(sym1)}
          val outputSym2 = if (einsumOutput.contains(sym2)) {einsumOutput(sym2)} else {Seq(sym2)}
          val einsumOutputVars = outputSym1 ++ outputSym2
          einsumOutput += (sym -> einsumOutputVars)
          // println(s"${sym.name} -> einsumOutputVars: ${einsumOutputVars}")
          var sym1Map = Map[Sym, Char]()
          var sym2Map = Map[Sym, Char]()
          if (einsumSymMap.contains(sym1)) {
            sym1Map = einsumSymMap(sym1)
          } else {
            sym1Map = createEinsumSymbolMap(Seq(sym1), einsumSymStart)
            einsumSymStart += 1 
          }

          if (einsumSymMap.contains(sym2)){
            sym2Map = einsumSymMap(sym2)
          } else {
            sym2Map = createEinsumSymbolMap(Seq(sym2), einsumSymStart)
            einsumSymStart += 1 
          }

          val symMap = sym1Map.++(sym2Map)
          // println(s"symMap: ${symMap}")
          einsumSymMap += (Sym(res1sym.name) -> symMap)
          // println(s"tensorList: ${tensorList}")
        case NormalizeQuery(sym1, dims) =>
          var newDims = Seq[Int]()
          for (d <- dims) {
            d match {
              case IntLiteral(i) =>
                newDims = newDims :+ i
              case _ => ??? // FIXME: Enable Exp: expression type
            }
          }
          val TensorLetN(bindings1, result1) = res
          var finalDim = TensorDim.Scalar
          if (!dims.isEmpty){ 
            finalDim = tensorList(sym1).dim
          }
          // println(s"symname: ${sym}")
          val res1sym = TensorSym(sym.name, finalDim)
          res = TensorLetN(bindings1 ++ Seq(res1sym -> TensorNormalize(tensorList(sym1), newDims)), res1sym)
          tensorList += (Sym(res1sym.name) -> res1sym)
          einsumOutputMap += (Sym(res1sym.name) -> einsumOutputMap(sym1))
          // println(s"tensorList: ${tensorList}")
          // println(s"einsumSymMap: ${einsumSymMap}")
          // println(s"einsumOutputMap: ${einsumOutputMap}")
          einsumSymMap += (Sym(res1sym.name) -> einsumSymMap(sym1))
          einsumOutput += (sym -> einsumOutput(sym1))
        case UnitMatrixQuery(v) =>
          val TensorLetN(bindings1, result1) = res
          // println(s"v: ${v}")
          v match {
            case UnitMatrix(sym1) => 
              val um = tensorList(sym1) // unitMatrix
              // println(s"tensorList: $tensorList")
              // println(s"um: ${um}")
              val res1sym = TensorSym(sym.name, um.dim)
              res = TensorLetN(bindings1 ++ Seq(res1sym -> TensorOnes(sym.name, um.dim)), res1sym)
              tensorList += (sym -> res1sym)
              unitMatriceTrack = unitMatriceTrack :+ sym
              // println(s"einsumSymMap: $einsumSymMap")
              // einsumOutputMap += (sym -> einsumOutputMap(sym1)) //FIXME: need to generalize
            case _ =>
              raise(s"Cannot handle `$query` yet.")
          }
        case LiftQuery(e) => 
          val TensorLetN(bindings1, result1) = res
          e match {
            case Lift(e1) => 
              val tenSym = tensorList(e1.asInstanceOf[Sym])
              val res1sym = TensorSym(s"${tenSym.name}_lift", tenSym.dim)
              res = TensorLetN(bindings1 ++ Seq(res1sym -> TensorLift(tenSym)), res1sym)
              tensorList += (sym -> res1sym)
            case _ => 
              raise(s"Cannot handle `$e` yet.")
          }
        case ReverseMapQuery(e, dicts) => 
          val TensorLetN(bindings1, result1) = res
          val finalDicts = dicts.map(d => tensorList(d.asInstanceOf[Sym])).toSeq
          val finalDim = TensorDim(dicts.size, finalDicts.map(d => d.dim.dims.last).toSeq)
          // println(s"finalDim: ${finalDim}")
          val res1sym = TensorSym(s"${e.asInstanceOf[Sym].name}_rev", finalDim)
          res = TensorLetN(bindings1 ++ Seq(res1sym -> TensorReverseMap(tensorList(e.asInstanceOf[Sym]), finalDicts, finalDim)), res1sym)
          tensorList += (sym -> res1sym)
        case DictionaryQuery(t) => 
          var bindings1 = Seq[(TensorSym, TensorExp)]()
          var result1 = TensorSym("null", TensorDim.Scalar).asInstanceOf[TensorExp]
          if (res != null) {
            val TensorLetN(bindings1_, result1_) = res
            bindings1 = bindings1_
            result1 = result1_
          }
          t match {
            case LoadDictionary(name, dim) =>
              val finalDim = TensorDim(dim.size, dim.map(x => x.asInstanceOf[IntLiteral].i).toSeq)
              val res1sym = TensorSym(name, finalDim)
              res = TensorLetN(bindings1 ++ Seq(res1sym -> DictionaryLoad(name, finalDim)), res1sym)
              tensorList += (sym -> res1sym)
            case _ => 
              raise(s"Cannot handle `$e` yet.")
          }
        case TensorQuery(t) =>
          // println(s"t: ${t}")
          // println(s"res: ${res}")
          var bindings1 = Seq[(TensorSym, TensorExp)]()
          var result1 = TensorSym("null", TensorDim.Scalar).asInstanceOf[TensorExp]
          if (res != null) {
            val TensorLetN(bindings1_, result1_) = res
            bindings1 = bindings1_
            result1 = result1_
          }
          // val TensorLetN(bindings1, result1) = res
          t match {
            case Filter(e: Sym, args) =>
              // println(s"e: ${e}")
              // println(s"tensorList: $tensorList")
              if (tensorList.contains(e)){ 
                val loadTensor = tensorList(e)
                // println(s"loadTensor.dim.dims.drop(0): ${loadTensor.dim.dims.drop(1)}")
                val finalDim = TensorDim(loadTensor.dim.order, Seq(args.size) ++ loadTensor.dim.dims.dropRight(1))
                // println(s"finalDim: $finalDim")
                val res1sym = TensorSym(loadTensor.name, finalDim) // Access at 0th dimension. Tensor order should not be changed.
                res = TensorLetN(bindings1 ++ Seq(res1sym -> TensorSelect(res1sym, args.map(x=>x.asInstanceOf[IntLiteral].i).toSeq, 0)), res1sym)
                tensorList += (sym -> res1sym)
              } else {
                throw new Exception(s"Tensor ${e.name} not loaded. Cannot access to this tensor.")
              }
            case LoadTensor(name, dim) =>
              val finalDim = TensorDim(dim.size, dim.map(x => x.asInstanceOf[IntLiteral].i).toSeq)
              val res1sym = TensorSym(name, finalDim)
              res = TensorLetN(bindings1 ++ Seq(res1sym -> TensorLoad(name, Seq(), finalDim)), res1sym)
              tensorList += (sym -> res1sym)
            case ToTensor(sym2: Sym) => 
              val finalDim = tensorList(sym2).dim
              val res1sym = TensorSym(sym.name, finalDim)
              val res2sym = TensorSym(sym2.name, finalDim)
              res = TensorLetN(bindings1 ++ Seq(res1sym -> res2sym), res1sym)
              tensorList += (sym -> res2sym)
            case IntLiteral(i) =>
              val res1sym = TensorSym(sym.name, TensorDim.Scalar)
              res = TensorLetN(bindings1 ++ Seq(res1sym -> TensorConst(Seq(i), TensorDim.Scalar)), res1sym)
              tensorList += (sym -> res1sym)
            case VectorLiteral(vec) =>
              val res1sym = TensorSym(sym.name, TensorDim.Vector(vec.size))
              res = TensorLetN(bindings1 ++ Seq(res1sym -> TensorConst(vec, TensorDim.Vector(vec.size))), res1sym)
              tensorList += (sym -> res1sym)
            case Sub(a: Sym, b: Sym) =>
              val tensorA = tensorList(a)
              val tensorB = tensorList(b)
              val finalDim = tensorA.dim
              val res1sym = TensorSym(sym.name, finalDim)
              // val equation = TensorBinOp(tensorA, tensorB, "-")
              val equation = TensorArithmetic(tensorA, tensorB, TensorMathOp.Subtract)
              res = TensorLetN(bindings1 ++ Seq(res1sym -> equation), res1sym)
              tensorList += (sym -> res1sym)
              // println(s"sym: ${sym}")
              // println(s"einsumOutputMap(a): ${einsumOutputMap(a)}")
              einsumOutputMap += (sym -> einsumOutputMap(a)) // slightly wrong
              einsumOutput += (sym -> einsumOutput(a)) // slightly wrong
            case Mult(a: Sym, b: Sym) =>
              val tensorA = tensorList(a)
              val tensorB = tensorList(b)
              def generateTensorSymAndEinsumSymbols(t: TensorSym): (String) = {
                var einsumSymbol = ""
                var counter = 0
                if (t.dim.order == 0) {
                  einsumSymbol = ""
                } else if (t.dim.order >= 0) {
                  for (tx <- t.dim.dims) {
                    einsumSymbol += ('a' + counter).toChar 
                    counter += 1
                  }
                } else {
                  raise(s"Invalid tensor `$a`.")
                }
                einsumSymbol
              }
              // println(s"tensorA: ${tensorA}")
              // println(s"tensorB: ${tensorB}")
              var aSymbol = generateTensorSymAndEinsumSymbols(tensorA)
              var bSymbol = generateTensorSymAndEinsumSymbols(tensorB)
              var optVar = tensorB
              var outputVarsSymbol = ""
              if (tensorA.dim.order <= tensorB.dim.order){
                outputVarsSymbol = bSymbol
              } else if (tensorA.dim.order > tensorB.dim.order) {
                optVar = tensorA
                outputVarsSymbol = aSymbol
              } else {
                raise(s"cannot handle multiplication of `$a` and `$b` yet.")
              }
              val finalDim = optVar.dim
              val res1sym = TensorSym(sym.name, finalDim)
              val args = Seq(tensorA, tensorB)
              val einsum = s"${aSymbol},${bSymbol}->${outputVarsSymbol}"
              // println(s"einsum: ${einsum}")
              res = TensorLetN(bindings1 ++ Seq(res1sym -> TensorContract(args, einsum, finalDim)), res1sym)
              tensorList += (sym -> res1sym)

              einsumOutputMap += (sym -> outputVarsSymbol)
              // einsumSymMap += (sym -> einsumSymMap(sym1))
              // if (einsumOutput.contains(a) && !einsumOutput.contains(b)) { // wrong, to be fixed
              if (einsumOutput.contains(a)) {
                einsumOutput += (sym -> einsumOutput(a))
              } else if (einsumOutput.contains(b)){
                einsumOutput += (sym -> einsumOutput(b))
              }
              
              // } else if (!einsumOutput.contains(a) && einsumOutput.contains(b)) {
              //   einsumOutput += (sym -> einsumOutput(b))
              // }
            case _ =>
              raise(s"cannot handle `$query` yet.")
          }
      }
    }
    res
  }

  def computeCPTs(e: Exp)(acc: Map[Sym, TensorExp]): Map[Sym, TensorExp] = e match {
    case LetBinding(x, e1 @ AcceptableBinding(_), e2) =>
      computeCPTs(e2)(acc + (x -> CPT.fromExp(e1)(acc.map(y => y._1 -> y._2.dim))))
    case LetBinding(x, QueryBinding(_), e2) =>
      computeCPTs(e2)(acc)
    case LetBinding(x, Query(_, _, _, _), e2) =>
      // FIXME specific to current case
      computeCPTs(e2)(acc + (x -> TensorSym(x.name, TensorDim(1, Seq(4)))))
    case _ =>
      acc
  }
  def varOrder(query: FullQuery): Seq[Sym] = {
    val dag = query.dag
    Graph.scc[Sym]((query.outputVars ++ query.obsSels.map(_.sym)).toList, x => dag(x).toList).flatten.reverse
  }
  // def varOrder(e: Exp): Seq[Sym] = e match {
  //   case LetBinding(x, AcceptableBinding(_), e2) =>
  //     x +: varOrder(e2)
  //   case LetBinding(x, QueryBinding(_), e2) =>
  //     varOrder(e2)
  //   case _ => Seq()
  // }
  def linearVE(query: FullQuery, vars: Seq[Sym], cpts: Map[Sym, TensorExp], einsumSymStart: Int): (Map[Sym, Char], String, TensorExp, Map[Sym, TensorSym]) = {
    // println(s"This is linear VE.")
    assert(query.outputVars.size == 1)
    assert(query.outputVars.head == vars.last)
    assert(query.obsSels.size == 0) // No update made to tensorList 
    val tensorVars = vars.map(v => v -> TensorSym(v.name, cpts(v).dim)).toMap
    var einsum = Map[Sym, Char]()
    var einsumOutput = ""
    val bindings = vars.map(v => {
      val pvars = query.dag(v)
      if(pvars.isEmpty) {
        tensorVars(v) -> cpts(v)
      } else {
        val ndag = pvars.map(pv => pv -> Seq()).toMap ++ Map(v -> query.dag(v))
        val tvs = tensorVars + (v -> cpts(v))
        einsum = createEinsumSymbolMap(pvars ++ Seq(v), einsumSymStart)
        val (body, outArgs) = contract(pvars ++ Seq(v), einsum, tvs, Seq(v), ndag)
        einsumOutput = outArgs
        tensorVars(v) -> body
      }
    })
    val tensorListUpdate = Map[Sym, TensorSym]()
    return (einsum, einsumOutput, TensorLetN(bindings, bindings.last._1), tensorListUpdate)
  }

  def createEinsumSymbolMap(vars: Seq[Sym], count: Int): Map[Sym, Char] = {
    val charVars = vars.zipWithIndex.map(vi => vi._1 -> ('a' + count + vi._2).toChar).toMap
    charVars
  }

  def contract(vars: Seq[Sym], charVars: Map[Sym, Char], tensorVars: Map[Sym, TensorExp], outputVars: Seq[Sym], dag: Dag[Sym]): (TensorExp, String) = {
    // if(linear)
    // println(s"contract($vars, $tensorVars, $outputVars)")
    // println(s"outputVars: ${outputVars}")
    // println(s"vars: ${vars}\n")
    // println(s"tensorVars: ${tensorVars.keySet}\n")
    // println(s"vars.map(tensorVars): ${vars.map(tensorVars)}\n")
    // println(s"outputVars: ${outputVars}\n")
    // println(s"tensorVars: ${tensorVars}\n")
    val n = outputVars.map(ov => tensorVars(ov).dim.dims(0)).toSeq
    // println(s"outputVars dims: ${}")
    val inArgs = vars.map(v => 
      if (dag.contains(v)) {
        s"${charVars(v)}${dag(v).map(charVars).mkString}"
      } else { // it is a weight
        charVars(v)
      }
    ) // for each tensor, einsum symbols: chd + parents
    val outArg = outputVars.map(charVars).mkString("")
    val einsumArg = s"${inArgs.mkString(",")}->$outArg"
    // println(s"einsumArg: ${einsumArg}")
    // println(s"n: $n")
    // println(s"tensorVars(Sym(lo_discount_weight)): ${tensorVars(Sym("lo_discount_weight"))}")
    return (TensorContract(vars.map(tensorVars), einsumArg, TensorDim(outputVars.size, n)), outArg)
  }
  def variableElimination(query: FullQuery, vars: Seq[Sym], cpts: Map[Sym, TensorExp], einsumSymStart: Int, tensorList: Map[Sym, TensorSym]): (Map[Sym, Char], String, TensorExp, Map[Sym, TensorSym]) = {
    // println(s"This is VE.")
    val outputVars = query.outputVars
    // println(s"query.obsSels.groupBy(_.sym): ${query.obsSels.groupBy(_.sym)}")
    val obssSels = query.obsSels.groupBy(_.sym).map(x => x._1 -> (x._2 match {
      case Seq(os) => os
      case list => raise(s"Only one obs|sel but found $list")
    })).toMap
    // println(s"obssSels: ${obssSels}")
    // println(s"query.weights: ${query.weights}")
    // println(s"cpts: ${cpts}")
    var tensorListUpdate = obssSels.map(
      x => x._2.sym -> TensorSym(x._2.sym.name, TensorDim(cpts(x._2.sym).dim.order, cpts(x._2.sym).dim.dims.dropRight(1) :+ x._2.values.size))
      ).toMap.asInstanceOf[Map[Sym, TensorSym]]
    // println(s"tensorListUpdate: ${tensorList}")
    var tensorVars = vars.map(v => if (obssSels.contains(v)) {
      v -> tensorListUpdate(v)
    } else {
      (v -> TensorSym(v.name, cpts(v).dim))
    }).toMap
    // println(s"tensorList: ${tensorList}")
    // println(s"tensorVars: ${tensorVars}")
    var einsumSymMap = createEinsumSymbolMap(vars, einsumSymStart)
    var vars2 = vars
    var tensorRhs = Map[Sym, TensorExp]()
    cpts.map(symTen => {
      var (sym, ten) = symTen
      val rhs = obssSels.get(sym) match {
        case Some(os) => TensorSelect(ten, os.values, 0) // -1
        case None => 
          ten match {
            case TensorLoad(_, args, _) =>
              if (args.length > 1) {
                raise("Doesn't support other sturctured BNs instead of chow-liu tree")
              } else {
                // println(s"args: ${args}")
                if (!args.isEmpty) {
                  if (obssSels.contains(Sym(args(0).name))) {
                    obssSels.get(Sym(args(0).name)) match {
                      case Some(os) => TensorSelect(ten, os.values, -1)
                      case None => ten
                    }
                    // TensorSelect(ten, (obssSels.get(args(0))).asInstanceOf[Seq[Int]])
                  } else {
                    ten
                  }
                } else {
                  ten
                }
              }
            case TensorSelect(_,_,_) | _ => ten
          }
      }
      tensorRhs += (sym -> rhs.asInstanceOf[TensorExp])

      // var frhs = rhs
      if (!query.weights.isEmpty) {
        val weightMap = query.weights.map(x => x.e1 -> x.e2).toMap
        // println(s"weightMap: ${weightMap}")
        if (weightMap.contains(sym)) {
          var filterWeight = Seq[TensorExp]()
          val weightSym = weightMap(sym)
          val tenSym = tensorList(weightSym)
          vars2 = vars2 :+ weightSym
          einsumSymMap += (weightSym -> einsumSymMap(sym))
          val weight_rhs = obssSels.get(sym) match {
            case Some(os) => 
              tensorVars += (weightSym -> TensorSym(tenSym.name, TensorDim(1, Seq(os.values.size)))) // not general
              tensorListUpdate += (weightSym -> TensorSym(tenSym.name, TensorDim(1, Seq(os.values.size))))
              TensorSelect(tenSym, os.values, 0)
            case None => 
              tensorVars += (weightSym -> tenSym)
              tenSym
          }
          tensorRhs += (weightSym -> weight_rhs.asInstanceOf[TensorExp])
          // println(s"filterWeight: ${filterWeight}")
          // frhs = TensorContract(Seq(rhs) ++ filterWeight, "ij,i->j", TensorDim(1, Seq(rhs.dim.dims.last))) // not general
        }
      }
    })

    // println(s"tensorRhs: ${tensorRhs2}")
    // println(s"tensorRhs: ${tensorRhs}")

    // val tensorRhs = cpts.map(symTen => {
    //   var (sym, ten) = symTen
    //   val rhs = obssSels.get(sym) match {
    //     case Some(os) => TensorSelect(ten, os.values, 0) // -1
    //     case None => 
    //       ten match {
    //         case TensorLoad(_, args, _) =>
    //           if (args.length > 1) {
    //             raise("Doesn't support other sturctured BNs instead of chow-liu tree")
    //           } else {
    //             // println(s"args: ${args}")
    //             if (!args.isEmpty) {
    //               if (obssSels.contains(Sym(args(0).name))) {
    //                 obssSels.get(Sym(args(0).name)) match {
    //                   case Some(os) => TensorSelect(ten, os.values, -1)
    //                   case None => ten
    //                 }
    //                 // TensorSelect(ten, (obssSels.get(args(0))).asInstanceOf[Seq[Int]])
    //               } else {
    //                 ten
    //               }
    //             } else {
    //               ten
    //             }
    //           }
    //         case TensorSelect(_,_,_) | _ => ten
    //       }
    //   }

    //   sym -> rhs.asInstanceOf[TensorExp]
    // })
    // println(s"tensorRhs: ${tensorRhs}")
    
    val (body, einsumOutput) = contract(vars2, einsumSymMap, tensorVars, outputVars, query.dag)
    // println(s"body: ${body}\n")
    // TODO implement using TensorLetN

    val res = vars2.foldRight[TensorExp](body)((cur, acc) => TensorLet(tensorVars(cur), tensorRhs(cur), acc))
    // println(s"res: ${res}")
    // println(backend.NumpyCodeGenerator.codegen(res))
    // println(query.dag)
    // println(query.dag.reverse)
    if(query.observations.nonEmpty && query.selections.nonEmpty)
      raise(s"Cannot handle lowering for both selections and observations")

    def newSym(name: String, e: TensorExp): TensorSym = 
      TensorSym(Sym.fresh(name).name, e.dim)

    def let(name: String)(e1: TensorExp)(f: TensorExp => TensorExp): TensorExp = e1 match {
      case TensorLetN(stmts, lastExpr) =>
        val nsym = newSym(name, lastExpr)
        TensorLetN(stmts :+ (nsym -> lastExpr), f(nsym))
      case lastExpr => 
        val nsym = newSym(name, lastExpr)
        TensorLet(nsym, lastExpr, f(nsym))
    }

    val resNormalized = 
      if(query.observations.nonEmpty) {
        let("normalized")(res)(x => TensorNormalize(x, Seq()))
      } else
        res
    let("res")(resNormalized)(x => x)
    return (einsumSymMap, einsumOutput, let("res")(resNormalized)(x => x), tensorListUpdate)
  }
}