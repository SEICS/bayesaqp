package bayestensor
package transformation

import frontend._
import ir._
import org.scalatest._
import Matchers._

class TensorInterpreterTest extends FlatSpec {
  def tensorEquality(e: TensorConst, expected: Array[Double]): Boolean = 
    e.elems.toArray.zip(expected).forall(x =>
      x._1 === x._2 +- 0.0001
    )

  implicit class TensorConstOps(e: TensorConst) {
    def =~=(expected: Array[Double]): Boolean = 
      e.elems.toArray.zip(expected).forall(x =>
        x._1 === x._2 +- 0.00001
      )
  }

  def readAndInterpret(filePath: String): TensorConst = {
    val texp = TensorSourceCode.fromFile(filePath).exp
    backend.TensorInterpreter.interpret(texp)(Map())
  }

  "TensorLowering" should "work for normalized inputs" in {
    val rexp1 = readAndInterpret("progs/tensor/obsProg_test1.ten")
    assert(rexp1 =~= Array(0.28,0.72))
    val rexp2 = readAndInterpret("progs/tensor/obsProg_test2.ten")
    assert(rexp2 =~= Array(0.28))
    val rexp3 = readAndInterpret("progs/tensor/obsProg_test3.ten")
    assert(rexp3 =~= Array(0.12,0.6))
    val rexp4 = readAndInterpret("progs/tensor/obsProg_test4.ten")
    assert(rexp4 =~= Array(0.28))
    // TODO requires a fix in the TensorDimChecker
    // val rexp5 = readAndInterpret("progs/tensor/count_2gb.ten")
    // assert(rexp5 =~= Array(0.099, 0.3025, 0.1485, 0.252, 0.099, 0.099))
  }
}