package bayestensor
package driver

import utils._
import frontend._
import backend._
import transformation._
import ir._
import java.io._

object Driver {
  sealed trait InputKind
  case object InputBN extends InputKind
  case class InputBaseline(ext: String) extends InputKind
  // var USE_SERVER = false
  var USE_SERVER = true
  val bns = Seq("insurance", "cancer", "hepar2", "alarm", "pigs", "survey", "water")
  val baselines = Seq("alarm", "evidence1", "evidence2", "grass", "noisyOr", 
                      "twoCoins", "murderMystery")
  def compile(filePath: String, kind: InputKind, backend: String): String = {
    val linear = time(kind match {
      case InputBN => true
      case InputBaseline(_) => false
    },"check if linear")
    val parsed = time(kind match {
      // case InputBaseline("ten") =>
      //   TensorSourceCode.fromFile(filePath).exp
      case InputBaseline(_) =>  
        SourceCode.fromFile(filePath).exp//,"parse btl file")
      case InputBN => 
        SourceCode.fromFile(filePath).exp//,"parse btl bn file")
    },"parse")
    // println(s"parsed: ${parsed}")
    val btExpr = time(kind match {
      case InputBaseline("bt") =>
        parsed
      // case InputBaseline("ten") => 
      //   parsed
      case _ => 
        parsed |>
        ObserveHoist |>
        DiceDesugar |>
        BayesTensorNormalizer 
    }, "optimize btExpr")
    val exp = time(btExpr |> new TensorLowering(linear), "TensorLowering btExpr")
    // println(s"exp: ${TensorCodeGenerator.codegen(exp)}")
    
    // // // // Inling
    val sexp = time(TensorToSetar(exp), "TensorToSetar")
    // println(s"sexp:\n${SetarCodeGenerator.codegen(sexp)}\n")
    // // // // val sexp_inlined = Inline(sexp)
    // // // // println(s"sexp_inlined:\n${SetarCodeGenerator.codegen(sexp_inlined)}\n")
    val sexp_opt = time(RewriteRules(sexp), "RewriteRules(Total)")
    // println(s"\nsexp_opt:\n${SetarCodeGenerator.codegen(sexp_opt)}\n")
    val texp = time(SetarToTensor(sexp_opt), "SetarToTensor")
    // val texp = exp // mircobenchmark - opt_phases: baseline
    // println(s"\ntexp:\n${TensorCodeGenerator.codegen(texp)}\n")
    // val npCode = 
    //   if(USE_SERVER) 
    //     time(NumpyCodeGenerator.apply(texp),"generateNumPy")
    //   else 
    //     time(NumpyCodeGenerator(texp),"generateNumPy")
    // npCode

    val genCode = if (backend == "numpy") {
      if(USE_SERVER) 
        time(NumpyCodeGenerator.apply(texp),"generateNumPy")
      else 
        time(NumpyCodeGenerator(texp),"generateNumPy")
    } else if (backend == "torch") {
      if(USE_SERVER) 
        time(TorchCodeGenerator.apply(texp),"generatePyTorch")
      else 
        time(TorchCodeGenerator(texp),"generatePyTorch")
    } else {
      raise(s"Backend ${backend} not supported.")
    }
    // println(s"genCode: ${genCode}")
    genCode
  }
  def writeToFile(outputFile: String, npCode: String): Unit = {
    val pw = new PrintWriter(outputFile)
    pw.println(npCode)
    pw.close()
  }
  def getFileName(name: String, kind: InputKind): String = 
    kind match {
      case InputBN => s"progs/dice/bns/$name.bif.dice"
      case InputBaseline(ext) => s"$name.$ext"//s"progs/dice/$name.$ext"
    }
      
  def generate_baseline(outputFolder: String, backend: String = "numpy"): Unit = {
    for(base <- baselines) {
      val kind = InputBaseline("dice")
      val filePath = getFileName(base, kind)
      val npCode = compile(filePath, kind, backend)
      val outputFile = s"${outputFolder}/${base}.py"
      writeToFile(outputFile, npCode)
    }
  }
  def generate_bn(outputFolder: String, backend: String = "numpy"): Unit = {
    for(bn <- bns) {
      val kind = InputBN
      val filePath = getFileName(bn, kind)
      val npCode = compile(filePath, kind, backend)
      val outputFile = s"${outputFolder}/${bn}.py"
      writeToFile(outputFile, npCode)
    }
  }
  def runPython(npCode: String): String = {
    if(USE_SERVER)
      PythonProxy.runPythonCodeServer(npCode)
    else
      PythonProxy.runPythonCode(npCode)
  }
  def e2e(file: String, kind: InputKind, backend: String = "numpy"): Unit = {
    val filePath = time(getFileName(file, kind),"getFileName")
    val genCode = time(kind match {
        case InputBaseline("ten") => 
          val exp = TensorSourceCode.fromFile(s"${file}.ten").exp
          val sexp = TensorToSetar(exp)
          val sexp_opt = RewriteRules(sexp)
          val texp = SetarToTensor(sexp_opt)
          val code = if (backend == "numpy") {
            NumpyCodeGenerator.apply(texp)
          } else if (backend == "torch") {
            TorchCodeGenerator.apply(texp)
          } else {
            raise(s"Backend ${backend} not supported.")
          }
          code
        case _ => 
          compile(filePath, kind, backend)
      }, "compile")
    
    // val npCode = time(kind match {
    //   case InputBaseline("ten") => 
    //     val exp = TensorSourceCode.fromFile(s"${file}.ten").exp
    //     // println(s"exp: ${exp}")
    //     val sexp = TensorToSetar(exp)
    //     // println(s"sexp:\n${SetarCodeGenerator.codegen(sexp)}\n")
    //     // val sexp_inlined = Inline(sexp)
    //     // println(s"sexp_inlined:\n${SetarCodeGenerator.codegen(sexp_inlined)}\n")
    //     val sexp_opt = RewriteRules(sexp)
    //     // println(s"sexp_opt:\n${SetarCodeGenerator.codegen(sexp_opt)}\n")
    //     val texp = SetarToTensor(sexp_opt)
    //     val npc = NumpyCodeGenerator.apply(texp)
    //     // println(s"npCode: ${npc}")
    //     npc
    //   case _ => 
    //     compile(filePath, kind)
    // }, "compile")
    // println(s"genCode: ${genCode}")

    // println(s"npcode: ${npCode} \nnpCode")
    // println(PythonProxy.runPythonCodeServer(npCode))
    // var lat = Seq[Double]()
    // for (i <- 0 until 8) { //8: warm up 3, averaged over 5 runs
    //   val pyRes = time(runPython(npCode), "python exec")
    //   // println(s"pyRes: ${pyRes}")
    //   if (i >= 3) {
    //     val pyTime = pyRes.split(",")(0).replace("(","").toDouble
    //     lat = lat :+ pyTime
    //   }
    // }
    // val avg_lat = lat.sum / lat.length
    // println(s"Average latency over ${lat.length} runs:${"%.2f".format(avg_lat)}ms.")

    // // // for nativeImage and hyperfine
    val pyRes = time(runPython(genCode), "python exec") 
    // // // val pyTime = pyRes.split(",")(0).replace("(","").toDouble
    // // // println(s"pyRes: ${pyRes}")
    // // // println(s"pyTime: ${pyRes(0)} pyRes: ${pyRes(1)}")
  }
  def parseFlag(args: String): Boolean = {
    if(args == "true") true else if(args == "false") false else raise(s"Illegal flag value: ${args}. Must be true or false.")
  }
  def main(args: Array[String]): Unit = {
//     PythonProxy.runPythonCodeServer("LOAD 'server/load_sample.py'")
//     val code = """from opt_einsum import contract
// def infer_customer_(wf_c_region,wf_c_city,wf_c_region_weight,wf_c_nation,wf_c_custkey):
//   probability = contract('ac,a,ch,ho,o->o', wf_c_region,wf_c_region_weight,wf_c_nation,wf_c_city,wf_c_custkey, optimize='dp')
//   return probability

// customer_ = infer_customer_(wf_c_region,wf_c_city,wf_c_region_weight,wf_c_nation,wf_c_custkey)
//     """
//     PythonProxy.runPythonCodeServer(code)
    if(args.length == 0) {
      generate_bn("generated/bns")
      generate_baseline("generated/baselines")
    } else if (args.length == 2) { // by default, use numpy
      val file = args(0)
      val kind = time(args(1) match {
        case "bn" => InputBN
        case "baseline" => InputBaseline("dice")
        case "bt" => InputBaseline("bt")
        case "tensor" => InputBaseline("ten")
        // case "inline" => InputBaseline("inline")
        case _ => raise(s"Not acceptable second arg: ${args.mkString(",")}")
      }, "findKind")
      time(e2e(file, kind, "numpy"), "e2e execution")
    } else if (args.length == 3) { // other backends, e.g., torch
      val file = args(0)
      val kind = time(args(1) match {
        case "bn" => InputBN
        case "baseline" => InputBaseline("dice")
        case "bt" => InputBaseline("bt")
        case "tensor" => InputBaseline("ten")
        // case "inline" => InputBaseline("inline")
        case _ => raise(s"Not acceptable second arg: ${args.mkString(",")}")
      }, "findKind")
      val backend = args(2)
      time(e2e(file, kind, backend), "e2e execution")
    } else {
      raise(s"Not acceptable args: ${args.mkString(",")}")
    }
  }
}