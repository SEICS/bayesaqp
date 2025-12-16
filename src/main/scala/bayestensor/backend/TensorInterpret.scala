package bayestensor
package backend

import ir._
import utils._

object TensorInterpreter {
  def interpret(e: TensorExp)(ctx: Map[TensorSym, TensorConst]): TensorConst = e match {
    case v @ TensorConst(vs, TensorDim(n, ds)) => v
    case TensorSelect(e1, ids, filterDim) => 
      val v = interpret(e1)(ctx)
      val dv = v.dim
      val dr = e.dim
      val N = dv.size
      val vs = v.elems.zipWithIndex.filter(xi => ids.contains(xi._2 % N)).map(_._1)
      // println(s"selected: $vs")
      TensorConst(vs, dr)
    case TensorLet(x, e1, e2) =>
      val v1 = interpret(e1)(ctx)
      interpret(e2)(ctx + (x -> v1))
    case v @ TensorSym(_, _) =>
      ctx(v)
    case TensorContract(args, einsum, dim) =>
      val vs = args.map(a => interpret(a)(ctx))
      val npCode = NumpyCodeGenerator.codegen(TensorContract(vs, einsum, dim))
      // println(npCode)
      // if(dim.order != 1) 
      //   raise(s"Tensor contraction doesn't support output order `${dim.order}`")
      val output = utils.PythonProxy.runPythonCode(s"import numpy as np; import opt_einsum; print($npCode)")
      // println(s"output: ${output}")
      val toutput = output.trim()
      // println(s"toutput: ${toutput}")
      try {
          // if(dim.order == 0)
          //   Seq(toutput.toDouble)
          // else
          //   toutput.substring(1, toutput.size - 1).trim.split(" ").filter(_.trim().size != 0).toSeq.map(_.toDouble)
        // val tres = toutput.split("array") // (inference_time, result)
        // val inference_time = tres(0).replace('[', ' ').replace(']', ' ').trim.split(" ").filter(_.trim().size != 0).toSeq.map(_.toDouble)
        // println(s"inference_time: ${inference_time}")
        val elems = toutput.replace('[', ' ').replace(']', ' ').trim.split(" ").filter(_.trim().size != 0).toSeq.map(_.toDouble)
        TensorConst(elems, dim)
      } catch {
        case ex => raise(s"Error in parsing `$toutput`")
      }
    case TensorArithmetic(e1, e2, op) =>
      // println(s"e1: ${e1}")
      val v1 = interpret(e1)(ctx)
      // println(s"v1: ${v1}")
      val v2 = interpret(e2)(ctx + (e1 -> v1))
      // println(s"v2: ${v2}")
      // print(s"op: ${op}")
      if (op.opStr == "*") { // FIXME: probably wrong but works for now.
        interpret(TensorContract(Seq(e1,e2),"i,i->i",e1.dim))(ctx + (e2 -> v2))
      } else if (op.opStr == "-") {
        // println(s"asd:${v2.elems.zipWithIndex.foreach(x => x._1 - v1.elems(x._2))}")
        TensorConst(v1.elems.zipWithIndex.map(x => x._1 - v2.elems(x._2)), e1.dim)
      } else if (op.opStr == "+") {
        TensorConst(v1.elems.zipWithIndex.map(x => x._1 + v2.elems(x._2)), e1.dim)
      } else if (op.opStr == "/") {
        TensorConst(v1.elems.zipWithIndex.map(x => x._1 / v2.elems(x._2)), e1.dim)
      } else {
        raise(s"Doesn't spport such arithmetic calculation of `$op`")
      }
      
    case TensorNormalize(e, normDims) => // FIXME
      val v = interpret(e)(ctx)
      // println(s"e: ${e}")
      // println(s"v: ${v}")
      // println(s"v.elems: ${v.elems}")
      
      val sum = v.elems.sum
      val elems = v.elems.map(x => x / sum)
      TensorConst(elems, v.dim)

      
      // val inArgs = v.dim.dims.zipWithIndex.map(x => 'a' + x._2).toSeq.mkString("")
      // val normArgs = (for (nd <- normDims) yield (inArgs.zipWithIndex.map(x => x._2 == nd)))
      // val outArgs = (for (nd <- normDims) yield (inArgs.zipWithIndex.map(x => x._2 != nd)))
      // val sum = TensorContract(Seq(v), s"${inArgs},${normArgs}->${outArgs}", v.dim)
      // TensorConst(v.elems / interpret(sum)(ctx).elems, v.dim)


      // val inputVars = e.dim.dims.zipWithIndex.map(vi => vi._2 -> ('a' + vi._2).toChar).toMap.values.mkString("")
      // val outputVars = if (!normDims.isEmpty){ 
      //   normDims.map(i => inputVars(i)).toSeq.mkString("")
      // } else {""}
      // val einsum = s"${inputVars}->${outputVars}" // "dy->y"
      // val finalDim = TensorDim(outputVars.length, normDims.map(dimX => e.dim.dims(dimX)).toSeq)
      // val normJoint = TensorContract(Seq(e), einsum, finalDim)
      // TensorConst((v/interpret(normJoint)(ctx)).elems, v.dim)
  }
}
