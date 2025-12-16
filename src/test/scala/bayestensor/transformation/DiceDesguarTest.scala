package bayestensor
package transformation

import frontend._
import ir._
import org.scalatest._
import Matchers._

class DiceDesugarTest extends FlatSpec {
  def readAndDesugar(filePath: String): Exp = {
    val exp = SourceCode.fromFile(filePath).exp
    val dexp = DiceDesugar(exp)
    // BayesTensorNormalChecker(dexp)
    dexp
  }

  "DiceDesugar" should "work for dice files" in {
    readAndDesugar("progs/dice/alarm.dice")
    readAndDesugar("progs/dice/evidence1.dice")
    readAndDesugar("progs/dice/evidence2.dice")
    readAndDesugar("progs/dice/grass.dice")
    readAndDesugar("progs/dice/noisyOr.dice")
    readAndDesugar("progs/dice/twoCoins.dice")
    readAndDesugar("progs/dice/obsProg.dice")
    readAndDesugar("progs/dice/bns/alarm.bif.dice")
    readAndDesugar("progs/dice/bns/cancer.bif.dice")
    readAndDesugar("progs/dice/bns/hepar2.bif.dice")
    readAndDesugar("progs/dice/bns/insurance.bif.dice")
    readAndDesugar("progs/dice/bns/pigs.bif.dice")
    readAndDesugar("progs/dice/bns/survey.bif.dice")
    readAndDesugar("progs/dice/bns/water.bif.dice")
  }
}