package bayestensor
package backend

import ir._
import setar._
import utils._

object SetarCodeGenerator {
  def codegen(prog: SetarProg): String = {
    val rd = prog.rules.map(codegenRule).mkDocument(doc"  # ")
    doc"${rd} # ${prog.ret.name}".toString
  }
  def codegenRule(rule: SetarRule): Document = {
    val where = 
      if(rule.filter.isEmpty) 
        doc""
      else 
        doc" where ${rcr(rule.filter)}"

    doc"${rcr(rule.hd)} = ${rcr(rule.body)}$where"
  }

  def rcr(e: SetarExp): Document = e match {
    case SetarAccess(ten, idcs) => doc"${rcr(ten)}(${idcs.map(_.name).mkString(", ")})"
    case SetarNorm(args, dims) => 
      val d = if(dims.isEmpty) doc"" else doc", ${dims.map(_.name).mkString(", ")}"
      doc"Norm(${rcr(args)}$d)"
    case SetarRange(idx, rng) => doc"(${idx.name} ∈ [${rng.map(_.toString).mkString(",")}])"
    case _ => doc"${e.toString} /* TODO */"
  }
  def rcr(e: SetarTensor): Document = e match {
    case SetarSym(name) => name
    // case SetarConstValue(vs, TensorDim(n, ds)) => 
    //   doc"tensor($n, [${ds.mkString(", ")}], [${vs.mkString(",")}])"
    case SetarWrapper(te) =>
      doc"@@${TensorCodeGenerator.rcr(te)}@@"
  }
  def rcr(e: SetarProduct): Document = e.map(rcr).mkDocument("*")
}
