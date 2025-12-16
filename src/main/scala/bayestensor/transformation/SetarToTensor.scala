package bayestensor
package transformation

import ir._
import utils._
import setar._
import frontend._
import backend._

object SetarToTensor extends (SetarProg => TensorExp) {
    def apply(e: SetarProg): TensorExp = {
        // println(s"e: ${SetarCodeGenerator.codegen(e)}")
        val (bindings, body) = e match { 
            case SetarProg(rules, ret) =>
                (rules.map(r => translateRule(r)), translateSym(ret))
            case _ => ???
        }
        
        // println(s"TensorCodeGenerator.codegen: ${TensorCodeGenerator.codegen(TensorLetN(bindings.flatten, body))}")
        TensorDimChecker(TensorLetN(bindings.flatten, body))
    }
    def idcsReverse(n: String): Int = n.toCharArray().map(n1 => n1.toInt - 97).toSeq(0) // Slightly wrong
    def translateSym(e: SetarSym): TensorSym = TensorSym(e.name, TensorDim.Scalar)
    def genEinsumTensorAndSymbol(sexp: SetarExp): (TensorSym, String) = {
        sexp match { // Tensor Contraction
            case SetarAccess(ten1, idcs) =>
                var einsumSymbols = Seq[String]()
                val ten2 = ten1 match {
                    case SetarSym(name) =>
                        TensorSym(name, TensorDim.Scalar)
                    case SetarWrapper(texp) =>
                        texp match {
                            case TensorOnes(name, dim) => 
                                TensorSym(name, dim)
                            case _ =>
                                 raise(s"Currently don't support `$sexp` for tensor contraction.")
                        }
                }
                val einsumSymbol = (idcs.map(i => i.name)).mkString("")
                (ten2, einsumSymbol)
            case _ => raise(s"Cannot generate tensor and symbol at TIL level for `$sexp`")
            }
    }

    def translateToTensorExp(body: SetarProduct, idcs: Seq[SetarIndex]): TensorExp = { // translateExp(body)
        if (body.length == 1) { // Selection, normalization, and etc.
            body(0) match {
                case SetarAccess(ten1, idcs1) =>
                    val ten2 = ten1 match {
                        case SetarWrapper(texp) => // TensorLoad, TensorConst, TensorOnes
                            texp
                        case SetarSym(name) => // TensorSym
                            TensorSym(name, TensorDim.Scalar)
                    }
                    
                    var texpf = ten2
                    val ifHiddenEinsum = !(idcs1.toSet.&~(idcs.toSet).isEmpty)
                    // println(s"ifHiddenEinsum;${ifHiddenEinsum}")
                    if (ifHiddenEinsum) {
                        val einsum = s"${idcs1.map(i => i.name).mkString("")}->${idcs.map(i => i.name).mkString("")}" // + "->" + idcs.map(i => i.name).mkString("")
                        texpf = TensorContract(Seq(ten2), einsum, TensorDim.Scalar)
                    }
                    // println(s"texpf: ${texpf}")
                    texpf
                   
                case SetarNorm(body, idcs2) => // Normalization
                    TensorNormalize(translateToTensorExp(body, idcs), idcs2.map(i => idcsReverse(i.name)).toSeq)
                case _ => 
                    raise(s"Transformation for `${body(0)}` from SetarExp to TensorExp is not yet supported.")
            }
        } else { // Tensor contraction 
            // println(s"body:\n${body}\n")
            var inArgs = Seq[TensorSym]()
            var einsum = Seq[String]()
            body.map(e => e match { // Tensor Contraction
                case s1: SetarAccess =>
                    val (arg, symbol) = genEinsumTensorAndSymbol(s1)
                    inArgs = inArgs :+ arg
                    einsum = einsum :+ symbol
                // case SetarNorm(args, idcs) => // TODO: miss implementation for "A(a,b) = Norm(B(c),...,D(b,a)) * K(a)"
                }
            )
            val outStr = idcs.map(i => i.name).mkString("")
            TensorContract(inArgs, einsum.mkString(",") + "->" + outStr, TensorDim.Scalar)
        }
    }

    // def translateToTensorExp(body: SetarProduct, idcs: Seq[SetarIndex]): TensorExp = {
    //     // println(s"translateExp(body): ${translateExp(body)}")
    //     translateExp(body, idcs) match {
    //         case TensorContract(args, einsum, dim) =>
    //             val outStr = idcs.map(i => i.name).mkString("")
    //             TensorContract(args, einsum + "->" + outStr, TensorDim.Scalar)
    //         case texp =>
    //             texp
    //     }
    // }

    def translateSelect(rhs: TensorExp, rule: SetarRule): TensorExp = {
        if (!rule.filter.isEmpty){ // selections
            val rhs1 = rule.filter(0) match { // slightly wrong
                case SetarRange(idx, rng) =>
                    TensorSelect(rhs, rng, idcsReverse(idx.name)) // Slightly wrong. Not general enough
                case _ => 
                    raise(s"Wrong SetarExp `${rule.filter}` for selections.")
            }
            rhs1
        } else {
            rhs
        }
    }

    def translateRule(rule: SetarRule): Seq[(TensorSym, TensorExp)] = {
        // println(s"rule: ${rule}")
        val ssym = translateSym(rule.hd.ten.asInstanceOf[SetarSym])
        var normCounter = 0
        var divCounter = 0
        var bindings = Seq[(TensorSym, TensorExp)]()
        if (rule.body.length == 1) {
            rule.body(0) match {
                case SetarAccess(ten1, idcs) =>
                    val rhs = translateToTensorExp(rule.body, rule.hd.idcs)
                        //  ten1 match {
                        // case SetarSym(name) => // TensorSym
                        //     if (rule.hd.idcs.size != idcs.size) { // This is a hidden einsum. e.g., 'ijk->k'
                        //         val args = Seq(TensorSym(name, TensorDim.Scalar))
                        //         val einsum = s"${idcs.map(i => i.name).mkString("")}->${rule.hd.idcs.map(j => j.name).mkString("")}"
                        //         TensorContract(args, einsum, TensorDim.Scalar)
                        //     } else {
                        //         translateToTensorExp(rule.body, rule.hd.idcs)
                        //     }
                        // case _ => translateToTensorExp(rule.body, rule.hd.idcs)
                    // }
                    bindings = bindings ++ Seq((ssym, translateSelect(rhs, rule)))

                case SetarDiv(numerator, nidcs, denominator, didcs) =>
                    val nsym = TensorSym(ssym.name + s"_divNum$divCounter", ssym.dim)
                    val nrhs = translateToTensorExp(numerator, nidcs)
                    bindings = bindings ++ Seq((nsym, translateSelect(nrhs, rule)))

                    val dsym = TensorSym(ssym.name + s"_divDen$divCounter", ssym.dim)
                    divCounter += 1
                    val drhs = translateToTensorExp(denominator, didcs)
                    bindings = bindings ++ Seq((dsym, translateSelect(drhs, rule)))

                    val frhs = TensorArithmetic(nsym, dsym, TensorMathOp.Divide)
                    bindings = bindings ++ Seq((ssym, frhs))
                case _ => 
                    val rhs = translateToTensorExp(rule.body, rule.hd.idcs)
                    bindings = bindings ++ Seq((ssym, translateSelect(rhs, rule)))
            }
        } else {
            var inArgs = Seq[TensorSym]()
            var einsum = Seq[String]()
            // println(s"rule.body: ${rule.body}")
            rule.body.zipWithIndex.map(b => 
                b._1 match {
                    case s1: SetarAccess =>
                        val (arg, symbol) = genEinsumTensorAndSymbol(s1)
                        inArgs = inArgs :+ arg
                        einsum = einsum :+ symbol
                    case SetarNorm(args, idcs) => // if norm in contraction, then create temp variable and replace Norm as temp variable in contraction
                        val nsym = TensorSym(ssym.name+s"_norm$normCounter", ssym.dim)
                        normCounter += 1
                        val nrhs = translateToTensorExp(Seq(b._1), rule.hd.idcs)
                        val nsymbol = (idcs.map(i => i.name)).mkString("")
                        bindings = bindings ++ Seq((nsym, translateSelect(nrhs, rule)))
                        inArgs = inArgs :+ nsym
                        einsum = einsum :+ nsymbol
                    case SetarDiv(numerator, nidcs, denominator, didcs) =>
                        val nsym = TensorSym(ssym.name + s"_divNum$divCounter", ssym.dim)
                        val nrhs = translateToTensorExp(numerator, nidcs)
                        val nsymbol = (nidcs.map(i => i.name)).mkString("")
                        bindings = bindings ++ Seq((nsym, translateSelect(nrhs, rule)))

                        val dsym = TensorSym(ssym.name + s"_divDen$divCounter", ssym.dim)
                        divCounter += 1
                        val drhs = translateToTensorExp(denominator, didcs)
                        val dsymbol = (didcs.map(i => i.name)).mkString("")
                        bindings = bindings ++ Seq((dsym, translateSelect(drhs, rule)))
                        
                        // division
                        val fsym = TensorSym(ssym.name + s"_div$divCounter", ssym.dim)
                        val frhs = TensorArithmetic(nsym, dsym, TensorMathOp.Divide)
                        val fsymbol = nsymbol
                        bindings = bindings ++ Seq((fsym, frhs))
                        inArgs = inArgs :+ fsym
                        einsum = einsum :+ fsymbol
                }
            )
            val outStr = rule.hd.idcs.map(i => i.name).mkString("")
            val texp = TensorContract(inArgs, einsum.mkString(",") + "->" + outStr, TensorDim.Scalar)
            // val rhs = translateToTensorExp(rule.body, rule.hd.idcs)
            // bindings = bindings ++ Seq((ssym, translateSelect(rhs, rule)))
            bindings = bindings ++ Seq((ssym, texp))
        }
        bindings
  }
}