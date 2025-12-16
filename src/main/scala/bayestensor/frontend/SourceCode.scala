package bayestensor
package frontend

import ir._
import transformation.PartialEvaluator

class SourceCode(val fileName: String, val exp: Exp)
class TensorSourceCode(val fileName: String, val exp: TensorExp)

object SourceCode {
  def fromFile(fileName: String): SourceCode = {
    val content = scala.io.Source.fromFile(fileName).mkString
    val prog = BayesTensorParser(content)
    new SourceCode(fileName, PartialEvaluator.inline(prog))
  }
}

object TensorSourceCode {
  def fromFile(fileName: String): TensorSourceCode = {
    val content = scala.io.Source.fromFile(fileName).mkString
    new TensorSourceCode(fileName, TensorParser(content))
  }
}