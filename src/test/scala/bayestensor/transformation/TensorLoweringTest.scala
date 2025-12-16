package bayestensor
package transformation

import frontend._
import ir._
import org.scalatest._
import Matchers._
import utils._

class TensorLoweringTest extends FlatSpec {
  val LONG_RUN = false
  implicit class TensorConstOps(e: TensorConst) {
    def =~=(expected: Array[Double]): Boolean = 
      e.elems.toArray.zip(expected).forall(x =>
        x._1 === x._2 +- 0.0001
      )
  }

  def readAndInterpret(filePath: String): TensorConst = {
    val exp = SourceCode.fromFile(filePath).exp
    val texp = new TensorLowering(false)(exp)
    // println(backend.TensorCodeGenerator.codegen(texp))
    backend.TensorInterpreter.interpret(texp)(Map())
  }

  def readNormalizeInterpret(filePath: String, linear: Boolean = false): TensorConst = {
    val exp = SourceCode.fromFile(filePath).exp
    val texp = 
      exp |> 
      ObserveHoist |>
      DiceDesugar |>
      BayesTensorNormalizer |>
      new TensorLowering(linear)
    // println(backend.TensorCodeGenerator.codegen(texp))
    backend.TensorInterpreter.interpret(texp)(Map())
  }

  "TensorLowering" should "work for dice inputs" in {
    val rexp1 = readNormalizeInterpret("progs/dice/alarm.dice")
    assert(rexp1 =~= Array(0.99700, 0.00299))
    val rexp2 = readNormalizeInterpret("progs/dice/evidence1.dice")
    // println(rexp2)
    assert(rexp2 =~= Array(0.66666, 0.33333))
    val rexp3 = readNormalizeInterpret("progs/dice/evidence2.dice")
    assert(rexp3 =~= Array(0.33333, 0.66666))
    val rexp4 = readNormalizeInterpret("progs/dice/grass.dice")
    assert(rexp4 =~= Array(0.29207, 0.70792))
    val rexp5 = readNormalizeInterpret("progs/dice/noisyOr.dice")
    assert(rexp5 =~= Array(0.18558, 0.81441))
    val rexp6 = readNormalizeInterpret("progs/dice/twoCoins.dice")
    assert(rexp6 =~= Array(0.66666, 0.33333))
    val rexp7 = readNormalizeInterpret("progs/dice/obsProg.dice")
    assert(rexp7 =~= Array(0.16666, 0.83333))
    val rexp8 = readNormalizeInterpret("progs/dice/murderMystery.dice")
    assert(rexp8 =~= Array(0.98418, 0.01581))
    val rexp9 = readNormalizeInterpret("progs/simple/murderMystery_inlined.bt")
    assert(rexp9 =~= Array(0.98418, 0.01581))
  }

  "TensorLowering" should "work for dice bns" in {
    val rexp1 = readNormalizeInterpret("progs/dice/bns/survey.bif.dice")
    // println(rexp1)
    assert(rexp1 =~= Array(0.56183398, 0.28085725, 0.15730877))
    val rexp2 = readNormalizeInterpret("progs/dice/bns/alarm.bif.dice", true) // doesn't terminate
    // println(rexp2)
    assert(rexp2 =~= Array(0.02721445, 0.2538232, 0.21101824, 0.5079441))
    val rexp3 = readNormalizeInterpret("progs/dice/bns/cancer.bif.dice")
    assert(rexp3 =~= Array(0.208141, 0.791859))
    val rexp5_1 = readNormalizeInterpret("progs/dice/bns/hepar2_test1.bif.dice", true)
    assert(rexp5_1 =~= Array(0.16269766, 0.83730253))
    if(LONG_RUN) {
      val rexp4 = readNormalizeInterpret("progs/dice/bns/pigs.bif.dice", true)
      assert(rexp4 =~= Array(0.25, 0.5, 0.25))
      val rexp5 = readNormalizeInterpret("progs/dice/bns/hepar2.bif.dice", true)
      assert(rexp5 =~= Array(0.42349967, 0.57649965))
      val rexp6 = readNormalizeInterpret("progs/dice/bns/insurance.bif.dice", true)
      assert(rexp6 =~= Array(0.47719294, 0.37424111, 0.1414326, 0.00713339)) // different answer
      // println(rexp6) 
      // DICE's output:
      // 0.562945586, 0.315187618, 0.10507771, 0.016789085
      val rexp7 = readNormalizeInterpret("progs/dice/bns/water.bif.dice", true)
      assert(rexp7 =~= Array(0.02640788, 0.82187436, 0.14731066, 0.0044021)) // different answer
      // println(rexp7)
      // DICE's output:
      // 0.028330451, 0.82139887, 0.142516259, 0.00775442
    }
  }

  "TensorLowering" should "work for normalized inputs" in {
    val rexp1 = readAndInterpret("progs/simple/alarm2_normalized.bt")
    // println(rexp1)
    assert(rexp1 =~= Array(0.80157,0.19842))
    val rexp2 = readAndInterpret("progs/simple/alarm3_normalized.bt")
    assert(rexp2 =~= Array(0.99890,0.00109))
    val rexp3 = readAndInterpret("progs/simple/obsProg_test4.bt")
    assert(rexp3 =~= Array(0.28))
    val rexp4 = readAndInterpret("progs/simple/noisyAnd2_normalized.bt")
    assert(rexp4 =~= Array(0.76,0.24))
    val rexp5 = readAndInterpret("progs/simple/noisyAnd3_normalized.bt")
    assert(rexp5 =~= Array(0.9040,0.0960))
    val rexp6 = readAndInterpret("progs/simple/alarm_normalized.bt")
    assert(rexp6 =~= Array(0.99700, 0.00299))
    val rexp7 = readAndInterpret("progs/simple/obsProg_normalized.bt")
    // println(rexp7)
    assert(rexp7 =~= Array(0.16666, 0.83333))
    // assert(rexp7 =~= Array(0.28))
  }

  "TensorLowering" should "work for toy approximate queries" in {
    val count_gb = readAndInterpret("progs/aqp/flights/toy/count_gb.bt")
    assert(count_gb =~= Array(5.033537240589E7, 4.869152589411E7))
    val avg = readAndInterpret("progs/aqp/flights/toy/avg.bt")
    assert(avg =~= Array(2632.3663398502254))
    val count_2gb_obs = readAndInterpret("progs/aqp/flights/toy/count_2gb_obs.bt")
    assert(count_2gb_obs =~= Array(0.14184444425933174, 0.12599231967479402, 0.19792248036185825, 0.3610585853873899, 0.041233850075387136, 0.13194832024123884))
    val count_2gb = readAndInterpret("progs/aqp/flights/toy/count_2gb.bt")
    assert(count_2gb =~= Array(0.099, 0.3025, 0.1485, 0.252, 0.099, 0.099))
    val avg_gb = readAndInterpret("progs/aqp/flights/toy/avg_gb.bt")
    assert(avg_gb =~= Array(1345.78984898, 1286.57649087))
    val sumA_noGB = readAndInterpret("progs/aqp/flights/toy/sumA.bt")
    // println(s"sumA_noGB: ${sumA_noGB}")
    assert(sumA_noGB =~= Array(1.5804677064359968E12)) // 1.580467706435996E12
    val sumAmultB_noGB = readAndInterpret("progs/aqp/flights/toy/sumAmultB.bt") // SUM(A*B) without group by
    // println(s"sumAmultB_noGB: ${sumAmultB_noGB}")
    assert(sumAmultB_noGB =~= Array(1.0182017378552282E15)) // 1.0182017378552286E15
    val sumA_sumB_noGB = readAndInterpret("progs/aqp/flights/toy/sumA_sumB.bt") // SUM(A)-SUM(B) without group by
    // println(s"sumA_sumB_noGB: ${sumA_sumB_noGB}")
    assert(sumA_sumB_noGB =~= Array(2.259454971636123E11))
    // val sumA_sumB_GB = readAndInterpret("progs/aqp/flights/toy/test.bt") // SUM(A)-SUM(B) with group by
    // println(s"sumA_sumB with GB: ${sumA_sumBs_GB}")
    // assert(sumA_sumB_GB =~= Array(1.15021964e+11, 1.10923537e+11))
  }
}