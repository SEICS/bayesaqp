package bayestensor
package backend

import ir._
import utils._

object TorchCodeGenerator extends (TensorExp => String) {
  val OPT_EINSUM = true
  def apply(e: TensorExp): String = 
    // Enable sharing intermediates' computations with opt_einsum
    s"""|t_start = perf_counter()
        |with shared_intermediates():
        |${codegen(e).split("\n").map(line => s"\t$line").mkString("\n")} 
        |t_end = perf_counter()
        |latency = (t_end-t_start)*1000
        |latency,force_cpu(${codegenRes(e)})""".stripMargin // force computed result to be returned from mps:0 device to cpu

  def codegenRes(e: TensorExp): String = e match {
    case TensorLetN(_, TensorSym(name, dim)) => s"${name}_"
    case _ => ""
  }

  def codegen(e: TensorExp): String = 
    rcr(e).toString

  def rcr(e: TensorExp): Document = e match {
    case TensorLet(x, e1, e2) =>
      doc"${x.name}_ = ${rcr(e1)} # ${rcr(e2)}"
    case TensorConst(vs, TensorDim(n, ds)) => 
      val arr = doc"torch.tensor([${vs.mkString(", ")}]).to(device)" // everything needs to be on the same device: mps/cpu
      if(n == 0)
        // vs(0).toString 
        doc"torch.tensor(${vs(0).toString}).to(device)" // Everything needs to be torch.tensor. Be careful: this is not the same as torch.Tensor
      else if(n == 1)
        arr
      else
        doc"$arr.reshape((${ds.mkString(",")}))"
    case TensorSelect(e, ids, filterDim) =>
      filterDim match {
        case _ =>
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
    case TensorLoad(name, _, dim) =>
      doc"${name}"
      // doc"torch_arr(${name})"
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
          doc", optimize='dp', use_blas=True, backend='torch'" // , order='C' flag 'dp' faster for Flights but slow for SSB // flag 'greedy' faster for higher-order tensor algebra (e.g., SSB's q11 (indexed from 0) will be 10x faster using 'greedy' than 'dp')
        else doc""
      // doc"$method('$einsum', ${args.map(rcr).map(i => s"densify_sparse(${i})").mkString(", ")}$opt_flag)" // no performance difference
      doc"$method('$einsum', ${args.map(rcr).mkDocument(", ")}$opt_flag)"
    case TensorArithmetic(e1, e2, op) =>
      // println(s"rcr(e1): ${op}")
      if (!"+*-/".contains(op.opStr)) {
        raise(s"Doesn't support math operator `$op`.")
      } 
      if (op.opStr == "/") { // to avoid producing Nans because of number/0
        doc"torch.where(${rcr(e2)}!=0, ${rcr(e1)} / ${rcr(e2)}, torch.zeros_like(${rcr(e1)}))" 
      } else {
        doc"${rcr(e1)} ${op.opStr} ${rcr(e2)}"
      }
    case TensorOnes(name, dim) =>
      doc"torch.ones((${dim.dims.mkString(",")}), dtype=torch.float32, device=device)"
    case TensorReverseMap(e, dicts, _) => 
      val dictList = dicts.map(d => s"${d.asInstanceOf[TensorSym].name}_").toSeq
      doc"reverse_map(${rcr(e)}.cpu().numpy(), ${dictList.mkString(", ")})" // cpu() because reverse_map expects numpy array
    case TensorLift(e) => 
      doc"np.array([k + (v,) for (k,v) in ${rcr(e)}.items()])" // ${rcr(e)}.reshape((${e.dim.dims.mkString(",")}))
    case _ => ???
  }
}