package bayestensor
package transformation

import ir._

object DiceDesugar extends (Exp => Exp) {
  object IfEqN {
    def unapply(e: Exp): Option[(Exp, Seq[(Int, Exp)])] = e match {
      case IfThenElse(Eq(e1, IntLiteral(i)), e3, e4) =>
        val nes = e4 match {
          case IfEqN(e1t, nes) if e1t == e1 =>
            nes
          case _ =>
            Seq((i + 1) -> e4)
        }
        Some(e1 -> (Seq(i -> e3) ++ nes))
      case _ => None
    }
  }
  def desugar(e: Exp): Exp = e match {
    case Flip(DoubleLiteral(p)) => Discrete(Seq(DoubleLiteral(1-p), DoubleLiteral(p)))
    case BoolLiteral(b) => IntLiteral(if(b) 1 else 0)
    case IfEqN(e1, es) =>
      MatchCase(desugar(e1), es.map(ie => IntLiteral(ie._1) -> desugar(ie._2)))
    case IfThenElse(e1, e2, e3) => MatchCase(desugar(e1), Seq(IntLiteral(1) -> desugar(e2), IntLiteral(0) -> desugar(e3)))
    case And(e1, e2) => desugar(IfThenElse(e1, e2, BoolLiteral(false)))
    case Or(e1, e2) => desugar(IfThenElse(e1, BoolLiteral(true), e2))
    case Imp(e1, e2) => desugar(IfThenElse(e1, e2, BoolLiteral(true)))
    case Not(e1) => desugar(IfThenElse(e1, BoolLiteral(false), BoolLiteral(true)))
    case Observe(e1, Seq()) => Observe(desugar(e1), Seq(IntLiteral(1)))
    case LetBinding(x, e1, e2) => LetBinding(x, desugar(e1), desugar(e2))
    case Discrete(_) => e
    case Sym(_) => e
    case _ => raise(s"DiceDesugar doesn't support $e")
  }
  def apply(e: Exp): Exp = desugar(e)
}