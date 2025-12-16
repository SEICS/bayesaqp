package bayestensor
package transformation

import frontend._
import ir._
import setar._
import backend._
import org.scalatest._
import Matchers._
import os._

class TensorToSetarTest extends FlatSpec {
  def readAndCompile(filePath: String): SetarProg = {
    val texp = TensorSourceCode.fromFile(filePath).exp
    // println(texp)
    val sexp = TensorToSetar(texp)
    // println(SetarCodeGenerator.codegen(sexp))
    // println(s"sexp: ${sexp}")

    // // Write each setar program into a file in directory: ~/Desktop/bayestensor/progs/setar/toTIL/...
    // val temp = filePath.split("/")
    // val setarFileName = temp.last.replace(".ten",".setar")
    // var basePath = os.Path("~/Desktop/bayestensor/progs/setar/toTIL")
    // val ssbPath = "progs/aqp/ssb/tensors/"
    // if (filePath contains ssbPath) {
    //   val idx = ssbPath.split("/").size
    //   if (idx != temp.size - 1) {
    //     if (temp(idx+2) == "opt" && temp(idx+1) != "no_gb") {
    //       basePath = basePath / temp(idx) / temp(idx+1) / temp(idx+2) / temp(idx+3)
    //     } else {
    //       basePath = basePath / temp(idx) / temp(idx+1) / temp(idx+2)
    //     }
        
    //   }
    // } else {
    //   basePath = basePath / "simple"
    // }
    // val setarFilePath: os.Path = basePath / setarFileName
    // os.write.over(setarFilePath, SetarCodeGenerator.codegen(sexp), createFolders = true)
    sexp
  }

  "TensorToSetar" should "work for normalized inputs" in {
    readAndCompile("progs/tensor/obsProg_test1.ten")
    readAndCompile("progs/tensor/obsProg_test2.ten")
    readAndCompile("progs/tensor/obsProg_test3.ten")
    readAndCompile("progs/tensor/obsProg_test4.ten")
    readAndCompile("progs/tensor/count_2gb.ten")
    readAndCompile("progs/tensor/scalar_contract.ten")
    readAndCompile("progs/tensor/scalar_contract2.ten")
  }

  "TensorToSetar" should "work for aqp tensors" in {
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