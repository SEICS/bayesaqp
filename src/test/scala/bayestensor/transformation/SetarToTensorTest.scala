package bayestensor
package transformation

import frontend._
import ir._
import setar._
import backend._
import org.scalatest._
import Matchers._
import os._

class SetarToTensorTest extends FlatSpec {
  def readAndCompile(filePath: String): Boolean = {
    println(s"Testing the file at: ${filePath}.\n")
    val texp = TensorSourceCode.fromFile(filePath).exp
    val sexp = TensorToSetar(texp)
    val texp2 = SetarToTensor(sexp)
    val sexp2 = TensorToSetar(texp2)
    sexp == sexp2
  }  

  "SetarToTensor" should "work for normalized inputs" in {
    readAndCompile("progs/tensor/obsProg_test1.ten")
    readAndCompile("progs/tensor/obsProg_test2.ten")
    readAndCompile("progs/tensor/obsProg_test3.ten")
    readAndCompile("progs/tensor/obsProg_test4.ten")
    readAndCompile("progs/tensor/count_2gb.ten")
    readAndCompile("progs/tensor/scalar_contract.ten")
    readAndCompile("progs/tensor/scalar_contract2.ten")
  }

  "SetarToTensor" should "work for aqp tensors" in {
    readAndCompile("progs/aqp/ssb/tensors/AVG_A_no_GB.ten")
    readAndCompile("progs/aqp/ssb/tensors/AVG_A_no_GB_alter.ten")
    readAndCompile("progs/aqp/ssb/tensors/AVG_A_no_GB_expect.ten")
    readAndCompile("progs/aqp/ssb/tensors/ssb_q0_avg_AmultB.ten")
    for(i <- 0 to 11) {
      readAndCompile(s"progs/aqp/ssb/tensors/sum/gb/opt/no_reverse_map/ssb_q${i}.ten")
      readAndCompile(s"progs/aqp/ssb/tensors/sum/gb/opt/reverse_map/ssb_q${i}.ten")
      readAndCompile(s"progs/aqp/ssb/tensors/sum/gb/opt/reverse_map/oneline_opt/ssb_q${i}.ten")
      readAndCompile(s"progs/aqp/ssb/tensors/sum/gb/ori/ssb_q${i}.ten")
      readAndCompile(s"progs/aqp/ssb/tensors/avg/gb/opt/no_reverse_map/ssb_q${i}.ten")
      readAndCompile(s"progs/aqp/ssb/tensors/avg/gb/opt/reverse_map/ssb_q${i}.ten")
      readAndCompile(s"progs/aqp/ssb/tensors/avg/gb/ori/ssb_q${i}.ten")
      readAndCompile(s"progs/aqp/ssb/tensors/avg/no_gb/ori/ssb-5gb_q${i}.ten")
      readAndCompile(s"progs/aqp/ssb/tensors/avg/no_gb/opt/ssb-5gb_q${i}.ten")
      println(s"*** $i ***")
    }
  }
}