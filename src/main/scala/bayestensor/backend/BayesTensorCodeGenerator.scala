package bayestensor
package backend

import ir._
import utils._

object BayesTensorCodeGenerator extends (Exp => String) {
  def apply(e: Exp): String = codegen(e)
  def codegen(e: Exp): String = 
    rcr(e).toString
  def rcrSeq(args: Seq[Exp]): Document = args.map(rcr).mkDocument(", ")
  def rcr(e: Exp): Document = e match {
    case BoolLiteral(b) => doc"${b.toString}"
    case IntLiteral(i) => doc"${i.toString}"
    case DoubleLiteral(d) => doc"${d.toString}"
    case BinOp(e1, e2, op) => doc"(${rcr(e1)} $op ${rcr(e2)})"
    case UnaryOp(e1, op) => doc"($op ${rcr(e1)})"
    case Sym(name) => doc"$name"
    case IfThenElse(e1, e2, e3) => doc"if(${rcr(e1)}) then ${rcr(e2)} else ${rcr(e3)}"
    case LetBinding(x, e1, e2) => doc"let ${x.name} = ${rcr(e1)} # ${rcr(e2)}"
    case MatchCase(e1, cs) => doc"""match (${rcr(e1)}) with  #{ 
${cs.map(x => doc"${rcr(x._1)} -> ${rcr(x._2)}").mkDocument(doc" # ")} #} """
    case Pair(e1, e2) => doc"(${rcr(e1)}, ${rcr(e2)})"
    case Fst(e1) => doc"(fst ${rcr(e1)})"
    case Snd(e1) => doc"(snd ${rcr(e1)})"
    case Observe(e1, args) => doc"observe ${rcr(e1)} [${rcrSeq(args)}]"
    case Discrete(args) => doc"discrete [${rcrSeq(args)}]"
    case Flip(e1) => doc"flip ${rcr(e1)}"
    case Select(e1, args) => doc"select ${rcr(e1)} [${rcrSeq(args)}]"
    case Query(vs, sels, obss, weights) => doc"query [${rcrSeq(vs)}] [${rcrSeq(sels)}] [${rcrSeq(obss)}] [${rcrSeq(weights)}]"
    // add expectation
  }
}
