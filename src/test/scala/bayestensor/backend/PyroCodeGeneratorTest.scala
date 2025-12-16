// package bayestensor
// package backend

// import frontend._
// import ir._
// import org.scalatest._
// import Matchers._

// class PyroCodeGeneratorTest extends FlatSpec {
//   def codegen(filePath: String): String = {
//     val exp = SourceCode.fromFile(filePath).exp
//     PyroCodeGenerator.codegen(exp)
//   }

//   "PyroCodeGenerator" should "work for normalized inputs" in {
//     val rexp1 = codegen("progs/simple/alarm2_normalized.bt")
//     // println(rexp1)
//     val rexp2 = codegen("progs/simple/noisyAnd2_normalized.bt")
//     // println(rexp2)
//     val rexp3 = codegen("progs/simple/noisyAnd3_normalized.bt")
//     // println(rexp3)
//     val rexp4 = codegen("progs/simple/alarm_normalized.bt")
//     // println(rexp4)
//     val rexp5 = codegen("progs/simple/obsProg_normalized.bt")
//     // println(rexp5)
//   }
// }