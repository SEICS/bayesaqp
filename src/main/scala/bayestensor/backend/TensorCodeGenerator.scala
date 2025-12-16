package bayestensor
package backend

import ir._
import utils._

object TensorCodeGenerator {
  def codegen(e: TensorExp): String = 
    rcr(e).toString
  def rcr(e: TensorExp): Document = e match {
    case TensorConst(vs, TensorDim(n, ds)) => 
      doc"tensor($n, [${ds.mkString(", ")}], [${vs.mkString(",")}])"
    case TensorSelect(e, ids, filterDim) =>
      doc"select(${rcr(e)}, [${ids.mkString(", ")}], ${filterDim})"
    case TensorLet(x, e1, e2) =>
      doc"let ${x.name} = ${rcr(e1)} in # ${rcr(e2)}"
    case TensorSym(name, _) =>
      doc"$name"
    case TensorNormalize(e1, normDims) =>
      doc"normalize(${rcr(e1)}, [${normDims.mkString(",")}])" // FIXME
    case TensorContract(args, einsum, _) =>
      doc"contract('$einsum', ${args.map(rcr).mkDocument(", ")})"
    case TensorArithmetic(e1, e2, op) =>
      doc"${rcr(e1)} ${op.opStr} ${rcr(e2)}"
    case TensorLoad(name, args, dim) =>
      doc"load('${name}', [${args.map(x => x.name).mkString(",")}], [${dim.dims.mkString(",")}])"
    case TensorOnes(name, dim) => 
      doc"ones(${name}, ${dim.order}, [${dim.dims.mkString(",")}])"
    case DictionaryLoad(name, dim) => 
      doc"load_dict('${name}', [${dim.dims.mkString(",")}])"
    case TensorReverseMap(e, dicts, dim) => 
      doc"map(${rcr(e)}, [${dicts.map(d => rcr(d)).mkString(",")}])"
    case TensorLift(e) => 
      doc"lift(${rcr(e)})"
  }
}
