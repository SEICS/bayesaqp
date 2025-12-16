package bayestensor
package frontend

import ir._
import org.scalatest._
import Matchers._

class BayesTensorParserTest extends FlatSpec {

  "BayesTensorParser" should "work for constant terms" in {
    bt"true" should be (BoolLiteral(true))
    bt"false" should be (BoolLiteral(false))
    bt"52" should be (IntLiteral(52))
    bt"-52" should be (IntLiteral(-52))
    bt"52.1" should be (DoubleLiteral(52.1))
    assertThrows[Exception] {
      bt""" "foo" """
    }
    assertThrows[Exception] {
      bt"`foo`"
    }
  }

  it should "work for if then else" in {
    bt"if true then 0 else 1" should be (IfThenElse(BoolLiteral(true), IntLiteral(0), IntLiteral(1)))
    bt"if (true) then (0) else (1)" should be (IfThenElse(BoolLiteral(true), IntLiteral(0), IntLiteral(1)))
    bt"if (!true) then (0) else (1)" should be (IfThenElse(Not(BoolLiteral(true)), IntLiteral(0), IntLiteral(1)))
  }

  it should "work for letbinding" in {
    bt"let x = 1 in 2" should be (LetBinding(Sym("x"), IntLiteral(1), IntLiteral(2)))
    bt"let    x  =    (1) in    2" should be (LetBinding(Sym("x"), IntLiteral(1), IntLiteral(2)))
    bt"let x_1 = 1 in 2" should be (LetBinding(Sym("x_1"), IntLiteral(1), IntLiteral(2)))
    bt"""let X = 1 in 
           2""" should be (LetBinding(Sym("X"), IntLiteral(1), IntLiteral(2)))
    bt"x__2" should be (Sym("x__2"))
    bt"X__2" should be (Sym("X__2"))
    assertThrows[Exception] {
      bt"let x = 1 (2)"
    }
  }

  it should "work for pairs" in {
    bt"(x, y)" should be (Pair(Sym("x"), Sym("y")))
    bt"fst x" should be (Fst(Sym("x")))
    bt"snd x" should be (Snd(Sym("x")))
  }

  it should "work for fun-call" in {
    bt"f(x)" should be (FunCall("f", Seq(Sym("x"))))
    bt"f(x, y)" should be (FunCall("f", Seq(Sym("x"), Sym("y"))))
    bt"f()" should be (FunCall("f", Seq()))
  }

  it should "work for pattern match" in {
    bt"match x with { 0 -> 42 }" should be (MatchCase(Sym("x"), Seq(IntLiteral(0) -> IntLiteral(42))))
    bt"match x with { 0 -> 42 | 1 -> 43 }" should be (MatchCase(Sym("x"), Seq(IntLiteral(0) -> IntLiteral(42), IntLiteral(1) -> IntLiteral(43))))
    bt"match (x, y) with { (0, 0) -> 42 | (1, 0) -> 43 }" should be (MatchCase(Pair(Sym("x"), Sym("y")), Seq(Pair(IntLiteral(0), IntLiteral(0)) -> IntLiteral(42), Pair(IntLiteral(1), IntLiteral(0)) -> IntLiteral(43))))
  }

  it should "work for observe/discrete/flip/select/vector/tensor" in {
    bt"observe x [1]" should be (Observe(Sym("x"), Seq(IntLiteral(1))))
    bt"observe x [1, 2, 3]" should be (Observe(Sym("x"), Seq(IntLiteral(1), IntLiteral(2), IntLiteral(3))))
    bt"select x [1]" should be (Select(Sym("x"), Seq(IntLiteral(1))))
    bt"select x [1, 2, 3]" should be (Select(Sym("x"), Seq(IntLiteral(1), IntLiteral(2), IntLiteral(3))))
    bt"discrete [0.1, 0.2, 0.3]" should be (Discrete(Seq(DoubleLiteral(0.1), DoubleLiteral(0.2), DoubleLiteral(0.3))))
    bt"observe x" should be (Observe(Sym("x"), Seq()))
    bt"flip 0.2" should be (Flip(DoubleLiteral(0.2)))
    bt"vector [0.2]" should be (VectorLiteral(Seq(0.2)))
    bt"vector [0.2,  0.4]" should be (VectorLiteral(Seq(0.2, 0.4)))
    bt"vector [0.2,  4]" should be (VectorLiteral(Seq(0.2, 4)))
    bt"tensor x" should be (ToTensor(Sym("x")))
    assertThrows[Exception] {
      bt"select x"
    }
    assertThrows[Exception] {
      bt"tensor [1.0]"
    }
  }

  it should "handle comments" in {
    val term = LetBinding(Sym("x"), Sym("y"), Sym("z"))
    bt"/* comment for let */ let x = y in z" should be (term)
    bt"let x = y in /* comment for let */ z" should be (term)
    bt"let x = y in z /* cc */" should be (term)
    bt"""// comment before
  let x = y in // comment middle 1
  // comment middle 2
  z
  """ should be (term)
  bt"""/* comment before */
  let x /* comment inside */ = /* comment inside */ y in 
  /* comment middle */
  z
  """ should be (term)
  }

  it should "work for arith" in {
    bt"2 * 3" should be (Mult(IntLiteral(2), IntLiteral(3)))
    bt"2 + 3" should be (Add(IntLiteral(2), IntLiteral(3)))
    bt"2 / 3" should be (Div(IntLiteral(2), IntLiteral(3)))
    bt"2 - 3" should be (Sub(IntLiteral(2), IntLiteral(3)))
    bt"2 + 1 * 3" should be (Add(IntLiteral(2), Mult(IntLiteral(1), IntLiteral(3))))
    bt"2 * 1 + 3" should be (Add(Mult(IntLiteral(2), IntLiteral(1)), IntLiteral(3)))
    bt"(2 * 1) + 3" should be (Add(Mult(IntLiteral(2), IntLiteral(1)), IntLiteral(3)))
    bt"-x + y" should be (Add(Neg(Sym("x")), Sym("y")))
    bt"-x * y" should be (Mult(Neg(Sym("x")), Sym("y")))
    bt"-(x + y)" should be (Neg(Add(Sym("x"), Sym("y"))))
    bt"2 < 3" should be (BinOp(IntLiteral(2), IntLiteral(3), "<"))
    bt"2 > 3" should be (BinOp(IntLiteral(2), IntLiteral(3), ">")) // FIXME
    bt"2 < 3 * 1" should be (BinOp(IntLiteral(2), Mult(IntLiteral(3), IntLiteral(1)), "<"))
    bt"2 < (3 * 1)" should be (BinOp(IntLiteral(2), Mult(IntLiteral(3), IntLiteral(1)), "<"))
    bt"2 * 3" should be (Mult(IntLiteral(2), IntLiteral(3)))
    bt"2 ^ 3" should be (Pow(IntLiteral(2), IntLiteral(3)))
    // // bt"2 ^ log(3)" should be (Pow(IntLiteral(2), Log(IntLiteral(3))))
    bt"2 && 3" should be (And(IntLiteral(2), IntLiteral(3)))
    bt"2 || 3" should be (Or(IntLiteral(2), IntLiteral(3)))
    bt"2 ~> 3" should be (Imp(IntLiteral(2), IntLiteral(3)))
    bt"!2" should be (Not(IntLiteral(2)))
  }

  it should "splice constants" in {
    val TRUE = true
    val FALSE = false
    val ONE = 1
    val ONE_HALF = 1.5
    bt"$TRUE" should be (bt"true")
    bt"$FALSE" should be (bt"false")
    bt"$ONE" should be (bt"1")
    bt"$ONE_HALF" should be (bt"1.5")
  }

  it should "normalize" in {
    bt"normalize prob []" should be (Normalize(Sym("prob"), Seq[IntLiteral]()))
  }

  it should "parse dice" in {
    SourceCode.fromFile("progs/dice/alarm.dice")
    SourceCode.fromFile("progs/dice/evidence1.dice")
    SourceCode.fromFile("progs/dice/evidence2.dice")
    SourceCode.fromFile("progs/dice/grass.dice")
    SourceCode.fromFile("progs/dice/murderMystery.dice")
    SourceCode.fromFile("progs/dice/noisyOr.dice")
    SourceCode.fromFile("progs/dice/twoCoins.dice")
    SourceCode.fromFile("progs/dice/obsProg.dice")
    SourceCode.fromFile("progs/dice/bns/alarm.bif.dice")
    SourceCode.fromFile("progs/dice/bns/cancer.bif.dice")
    SourceCode.fromFile("progs/dice/bns/hepar2.bif.dice")
    SourceCode.fromFile("progs/dice/bns/insurance.bif.dice")
    SourceCode.fromFile("progs/dice/bns/pigs.bif.dice")
    SourceCode.fromFile("progs/dice/bns/survey.bif.dice")
    SourceCode.fromFile("progs/dice/bns/water.bif.dice")
  }


  it should "parse simple" in {
    SourceCode.fromFile("progs/simple/alarm_tensorized.bt")
    SourceCode.fromFile("progs/simple/alarm_normalized.bt")
    SourceCode.fromFile("progs/simple/alarm2_normalized.bt")
    SourceCode.fromFile("progs/simple/alarm3_normalized.bt")
    SourceCode.fromFile("progs/simple/obsProg_normalized.bt")
    SourceCode.fromFile("progs/simple/murderMystery_inlined.bt")
  }

  it should "parse aqp" in {
    SourceCode.fromFile("progs/aqp/ssb/ssb_simple1.bt")
    SourceCode.fromFile("progs/aqp/ssb/ssb_simple2.bt")
    SourceCode.fromFile("progs/aqp/ssb/ssb_simple3.bt")
    SourceCode.fromFile("progs/aqp/ssb/ssb_simple4.bt")
    SourceCode.fromFile("progs/aqp/ssb/ssb_simple5.bt")
    SourceCode.fromFile("progs/aqp/ssb/ssb_simple6.bt")
    SourceCode.fromFile("progs/aqp/flights/toy/count_gb.bt")
    SourceCode.fromFile("progs/aqp/flights/toy/count_2gb.bt")
    SourceCode.fromFile("progs/aqp/flights/toy/count_2gb_obs.bt")
    SourceCode.fromFile("progs/aqp/flights/toy/avg.bt")
    SourceCode.fromFile("progs/aqp/flights/toy/avg_gb.bt")
    SourceCode.fromFile("progs/aqp/flights/toy/sumA.bt")
    SourceCode.fromFile("progs/aqp/flights/toy/avg_load.bt")
    // SourceCode.fromFile("progs/aqp/flights/toy/sum_gb.bt")
  }
}