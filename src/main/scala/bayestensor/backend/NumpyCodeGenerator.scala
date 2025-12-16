package bayestensor
package backend

import ir._
import utils._

object NumpyCodeGenerator extends (TensorExp => String) {
  val OPT_EINSUM = true
  def apply(e: TensorExp): String = 
    // Enable sharing intermediates' computations with opt_einsum
    s"""|t_start = perf_counter()
        |with shared_intermediates():
        |${codegen(e).split("\n").map(line => s"\t$line").mkString("\n")} 
        |t_end = perf_counter()
        |latency = (t_end-t_start)*1000
        |latency,${codegenRes(e)}""".stripMargin

  def codegenRes(e: TensorExp): String = e match {
    case TensorLetN(_, TensorSym(name, dim)) => s"${name}_"
    case _ => ""
  }

  def codegen(e: TensorExp): String = 
    rcr(e).toString

  def rcr(e: TensorExp): Document = e match {
    case TensorConst(vs, TensorDim(n, ds)) => 
      val arr = doc"np.array([${vs.mkString(", ")}])"
      if(n == 0)
        vs(0).toString
      else if(n == 1)
        arr
      else
        // doc"$arr"
        doc"$arr.reshape((${ds.mkString(",")}))"
        // doc"densify_sparse($arr)"
    case TensorSelect(e, ids, filterDim) =>
      filterDim match {
        // case 0 => doc"${rcr(e)}[[${ids.mkString(",")}]]" // same efficiency as below
        case _ =>
          // doc"${rcr(e)}[${":,"* (filterDim - 1)} [${ids.mkString(",")}]]" // same efficiency as below
          // doc"np.take(${rcr(e)}, [${ids.mkString(",")}], axis=${filterDim})" // doesn't work for sparse tensor
          doc"take_tensor(${rcr(e)}, [${ids.mkString(",")}], axis=${filterDim})"
      }
    case TensorNormalize(e, normDims) =>
      val inputVars = e.dim.dims.zipWithIndex.map(vi => vi._2 -> ('a' + vi._2).toChar).toMap.values.mkString("")
      val outputVars = if (!normDims.isEmpty){ 
        normDims.map(i => inputVars(i)).toSeq.mkString("")
      } else {""}
      val einsum = s"${inputVars}->${outputVars}" // "dy->y"
      val finalDim = TensorDim(outputVars.length, normDims.map(dimX => e.dim.dims(dimX)).toSeq)
      val normJoint = TensorContract(Seq(e), einsum, finalDim)
      doc"${rcr(e)} / ${rcr(normJoint)}" // shape to divide must match
    case TensorLet(x, e1, e2) =>
      doc"${x.name}_ = ${rcr(e1)} # ${rcr(e2)}"
    case TensorLoad(name, _, dim) =>
      // doc"${name}.reshape((${dim.dims.mkString(",")}))"
      // doc"densify_sparse($name)"
      doc"${name}"
    case TensorSym(name, _) =>
      doc"${name}_"
    case DictionaryLoad(name, _) => 
      doc"${name}.flat[0]"
    case TensorContract(args, einsum, _) =>
      val method = 
        if(OPT_EINSUM) 
          doc"contract" // contract_expression can be cached with shared_intermediates():
        else
          doc"np.einsum"
      val opt_flag = 
        if (OPT_EINSUM)
          doc", optimize='dp', use_blas=True" // , order='C' flag 'dp' faster for Flights but slow for SSB // flag 'greedy' faster for higher-order tensor algebra (e.g., SSB's q11 (indexed from 0) will be 10x faster using 'greedy' than 'dp')
        else doc""
      // doc"$method('$einsum', ${args.map(rcr).map(i => s"densify_sparse(${i})").mkString(", ")}$opt_flag)" // no performance difference
      doc"$method('$einsum', ${args.map(rcr).mkDocument(", ")}$opt_flag)"
    case TensorArithmetic(e1, e2, op) =>
      // println(s"rcr(e1): ${op}")
      if (!"+*-/".contains(op.opStr)) {
        raise(s"Doesn't support math operator `$op`.")
      } 
      if (op.opStr == "/") { // to avoid producing Nans because of number/0
        doc"np.divide(${rcr(e1)}, ${rcr(e2)}, out=np.zeros_like(${rcr(e1)}), where=${rcr(e2)}!=0)"
      } else {
        doc"${rcr(e1)} ${op.opStr} ${rcr(e2)}"
      }
    case TensorOnes(name, dim) =>
      doc"np.ones((${dim.dims.mkString(",")}))"
    case TensorReverseMap(e, dicts, _) => 
      val dictList = dicts.map(d => s"${d.asInstanceOf[TensorSym].name}_").toSeq
      doc"reverse_map(${rcr(e)}, ${dictList.mkString(", ")})"
    case TensorLift(e) => 
      doc"np.array([k + (v,) for (k,v) in ${rcr(e)}.items()])" // ${rcr(e)}.reshape((${e.dim.dims.mkString(",")}))
    case _ => ???
  }
}
