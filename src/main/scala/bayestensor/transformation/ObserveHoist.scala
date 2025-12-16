package bayestensor
package transformation

import ir._

object ObserveHoist extends (Exp => Exp) {
  def hoistOneLevel(e: Exp, cond: Exp): Exp = e match {
    // case LetBinding(x, e1, e2) => raise(s"Doesn't handle hoistOneLevel for $e")
    case Observe(e1, vs) => Observe(Imp(cond, e1), vs)
    case _ => e
  }
  def hoist(e: Exp)(inIf: Boolean): (Exp, Seq[(Sym, Exp)]) = e match {
    case IfThenElse(e1, e2, e3) => 
      val (te2, se2) = hoist(e2)(true)
      val (te3, se3) = hoist(e3)(true)
      val binds = se2.map(x => x._1 -> hoistOneLevel(x._2, e1)) ++ se3.map(x => x._1 -> hoistOneLevel(x._2, Not(e1)))
      if(!inIf) {
        LetBindingN(
          binds, 
          IfThenElse(e1, te2, te3)) -> Seq()
      } else
        IfThenElse(e1, te2, te3) -> binds
    case LetBinding(x, e1, e2) =>
      val (te1, se1) = hoist(e1)(inIf)
      val (te2, se2) = hoist(e2)(inIf)
      if(!inIf) {
        LetBinding(x, te1, te2) -> (se1 ++ se2)
      } else {
        te2 -> (se1 ++ Seq(x -> te1) ++ se2 )
      }
    // case Discrete(vs) =>
    case _ => 
      e -> Seq()
  }
  def apply(e: Exp): Exp = {
    val (te, se) = hoist(e)(false)
    assert(se.isEmpty)
    PartialEvaluator.simplify(Map())(te)
  }
}