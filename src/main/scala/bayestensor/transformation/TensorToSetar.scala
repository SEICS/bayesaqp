package bayestensor
package transformation

import ir._
import utils._
import setar._
import frontend._

object TensorToSetar extends (TensorExp => SetarProg) {
  def apply(e: TensorExp): SetarProg = {
    // println(s"e: ${e}\n")
    val (rules, ret) = TensorDimChecker(e) match {
      case TensorLetN(stmts, ret: TensorSym) =>
        (stmts.map(st => translateStmt(st._1, st._2)), translateSym(ret))
      case _ => ???
    }
    SetarProg(rules, ret)
  }
  val IDX_LIST = (0 to 28).map(i => (i + 'a').toChar)
  def genIdcs(n: Int) = IDX_LIST.take(n).map(char => SetarIndex(char.toString))
  def translateSym(e: TensorSym): SetarSym = SetarSym(e.name)
  def translateExp(e: TensorExp): (SetarProduct, SetarProduct, Seq[SetarIndex]) = 
    e match {
      // case TensorConst(vs, dim @ TensorDim(n, ds))  =>
      //   val idcs = genIdcs(n)
      //   (Seq(SetarAccess(SetarWrapper(e), idcs)), Seq(), idcs)
      case TensorContract(args, einsum, dim) =>
        val temp_str = einsum.filterNot(_.isWhitespace).split("->") // Remove all space from the einsum expression
        val istrs = (temp_str(0)).split(",", -1).toSeq // Includes empty string: e.g., "a,b,," => ("a", "b", "", "")
        assert(istrs.size == args.size)
        val ostr = if (temp_str.size == 2) temp_str(1) else ""
        def strToIdcs(str: String): Seq[SetarIndex] = 
          str.map(c => SetarIndex(c.toString))
        val idcs = strToIdcs(ostr)
        val body = args.zip(istrs).map(ai => {
          if(ai._1.dim.order != (ai._2.size - ai._2.count(_ == '.'))) {
            raise(s"contraction `$e` isn't valid: ${ai._1.dim.order} != ${ai._2.size} for `$ai`")
          }
          SetarAccess(translateSym(ai._1.asInstanceOf[TensorSym]), strToIdcs(ai._2))
        })
        (body, Seq(), idcs)
      case sym @ TensorSym(name, dim) => 
        val idcs = genIdcs(e.dim.order)
        val body = SetarAccess(translateSym(sym), idcs)
        (Seq(body), Seq(), idcs)
      case sel @ TensorSelect(se, ids, _) =>
        val (body, filter, idcs) = translateExp(se)
        (body, filter ++ Seq(SetarRange(idcs(sel.actualFilterDim), ids)), idcs)
      case TensorNormalize(ne, dims) =>
        val (body, filter, idcs) = translateExp(ne)
        (Seq(SetarNorm(body, dims.map(d => idcs(d)))), filter, idcs)
      case _: TensorLoad | _: TensorConst | _: TensorOnes | _: DictionaryLoad | _: TensorLift | _: TensorReverseMap | _: TensorArithmetic =>
        val idcs = genIdcs(e.dim.order)
        (Seq(SetarAccess(SetarWrapper(e), idcs)), Seq(), idcs)
      case _ =>
        raise(s"tensor expression $e cannot be lifted to Setar")
    }
    
  def translateStmt(sym: TensorSym, rhs: TensorExp): SetarRule = {
    val ssym = translateSym(sym)
    val (body, filter, idcs) = translateExp(rhs)
    val hd = SetarAccess(ssym, idcs)
    SetarRule(hd, body, filter)
  }
}