package bayestensor
package transformation

import frontend._
import ir._
import org.scalatest._
import Matchers._

class BayesTensorNormalizerTest extends FlatSpec {
  def readAndNormalize(filePath: String): Exp = {
    val exp = SourceCode.fromFile(filePath).exp
    val hexp = ObserveHoist(exp)
    // println(backend.BayesTensorCodeGenerator(hexp))
    // println("----")
    val dexp = DiceDesugar(hexp)
    // println(backend.BayesTensorCodeGenerator(dexp))
    val nexp = BayesTensorNormalizer(dexp)
    // println(backend.BayesTensorCodeGenerator(nexp))
    BayesTensorNormalChecker(nexp)
    nexp
  }

  "BayesTensorNormalizer" should "work for dice files" in {
    readAndNormalize("progs/dice/alarm.dice")
    readAndNormalize("progs/dice/evidence1.dice")
    readAndNormalize("progs/dice/evidence2.dice")
    readAndNormalize("progs/dice/grass.dice")
    readAndNormalize("progs/dice/noisyOr.dice")
    readAndNormalize("progs/dice/twoCoins.dice")
    readAndNormalize("progs/dice/obsProg.dice")
    readAndNormalize("progs/dice/murderMystery.dice")
    readAndNormalize("progs/dice/bns/alarm.bif.dice")
    readAndNormalize("progs/dice/bns/alarm_test1.bif.dice")
    readAndNormalize("progs/dice/bns/cancer.bif.dice")
    readAndNormalize("progs/dice/bns/hepar2.bif.dice")
    readAndNormalize("progs/dice/bns/hepar2_test1.bif.dice")
    readAndNormalize("progs/dice/bns/insurance.bif.dice")
    readAndNormalize("progs/dice/bns/pigs.bif.dice")
    readAndNormalize("progs/dice/bns/survey.bif.dice")
    readAndNormalize("progs/dice/bns/water.bif.dice")
  }
}