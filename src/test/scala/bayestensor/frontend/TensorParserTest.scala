package bayestensor
package frontend

import ir._
import org.scalatest._
import Matchers._

class TensorParserTest extends FlatSpec {

  "TensorParser" should "work for constant terms" in {
    tensor"tensor(1, [2], [0, 1])" should be (TensorConst(Seq(0, 1.0), TensorDim(1, Seq(2))))
    tensor"tensor(2, [2, 2], [0, 1, 0.2, 0.08])" should be (TensorConst(Seq(0, 1.0, 0.2, 0.08), TensorDim(2, Seq(2, 2))))
    assertThrows[Exception] {
      tensor"tensor(2, [2], [0, 1])"
    }
    assertThrows[Exception] {
      tensor"tensor(1, [2], [0, 1, 1.5])"
    }
    assertThrows[Exception] {
      tensor"tensor(2, [2, 2], [0, 1, 0, 1, 2])"
    }
  }

  it should "work for letbinding" in {
    val xDim = TensorDim(1, Seq(2))
    val xVar = TensorSym("x", xDim)
    tensor"let x = tensor(1, [2], [0, 1]) in x" should be (TensorLet(xVar, TensorConst(Seq(0, 1), xDim), xVar))
    assertThrows[Exception] {
      tensor"let x = 1 (2)"
    }
  }

  it should "work for contraction" in {
    val xDim = TensorDim(1, Seq(2))
    val yDim = TensorDim(2, Seq(2, 2))
    val xVar = TensorSym("x", xDim)
    val yVar = TensorSym("y", yDim)
    tensor"let x = tensor(1, [2], [0, 1]) in let y = tensor(2, [2, 2], [0, 1, 0, 1]) in contract('i,ij->j', x, y)" should be (
      TensorLet(xVar, TensorConst(Seq(0, 1), xDim), 
        TensorLet(yVar, TensorConst(Seq(0, 1, 0, 1), yDim), 
        TensorContract(Seq(xVar, yVar), "i,ij->j", xDim))))
    assertThrows[Exception] {
      tensor"contract('i,ij->j', x, y)"
    }
  }

  it should "work for select" in {
    val xDim = TensorDim(1, Seq(2))
    val xVar = TensorSym("x", xDim)
    tensor"let x = tensor(1, [2], [0, 1]) in select(x, [1], -1)" should be (
      TensorLet(xVar, TensorConst(Seq(0, 1), xDim), 
        TensorSelect(xVar, Seq(1), -1))
    )
  }

  it should "work for normalize" in {
    val xDim = TensorDim(1, Seq(2))
    val xVar = TensorSym("x", xDim)
    tensor"let x = tensor(1, [2], [0, 1]) in normalize(x, [ 1 ])" should be (
      TensorLet(xVar, TensorConst(Seq(0, 1), xDim), 
        TensorNormalize(xVar, Seq(1)))
    )
    tensor"let x = tensor(1, [2], [0, 1]) in normalize(x, [])" should be (
      TensorLet(xVar, TensorConst(Seq(0, 1), xDim), 
        TensorNormalize(xVar, Seq()))
    )
  }

  it should "work for arith" in {
    tensor"x + y" should be (TensorAdd(TensorSym("x", TensorDim.Scalar), TensorSym("y", TensorDim.Scalar)))
    tensor"x / y" should be (TensorDiv(TensorSym("x", TensorDim.Scalar), TensorSym("y", TensorDim.Scalar)))
    tensor"x - y" should be (TensorSub(TensorSym("x", TensorDim.Scalar), TensorSym("y", TensorDim.Scalar)))
  }

  it should "parse tensor files" in {
    TensorSourceCode.fromFile("progs/tensor/obsProg_test1.ten")
    TensorSourceCode.fromFile("progs/tensor/obsProg_test2.ten")
    TensorSourceCode.fromFile("progs/tensor/obsProg_test3.ten")
  }
}