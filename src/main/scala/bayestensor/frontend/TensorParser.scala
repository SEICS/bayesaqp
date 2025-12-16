package bayestensor
package frontend

import ir._
import fastparse._, NoWhitespace._, CharPredicates._


object TensorParser {
  def keywords[_: P] = P (
    StringIn("tensor", "select", "let", "in", "contract", "normalize", "load", "ones") ~
      !idRest
  )  

  def singleComment[_: P] = P( "//" ~/ (!newline ~ AnyChar).rep ~/ newline)
  def multiComment[_: P]  = P( "/*" ~/ (!"*/" ~ AnyChar).rep ~/ "*/")
  def comment[_: P]       = P( multiComment | singleComment )

  def newline[_: P]       = P( "\n" | "\r\n" | "\r" )
  def whitespace[_: P]    = P( " " | "\t" | newline)
  def spaceToken[_: P]    = P( comment | whitespace.rep(1) )
  def space[_: P]         = P( spaceToken.rep )
  def digits[_: P]        = P( CharsWhileIn("0-9") )
  def exponent[_: P]      = P( CharIn("eE") ~ CharIn("+\\-").? ~ digits )
  def fractional[_: P]    = P( "." ~ digits )
  def integral[_: P]      = P( "0" | CharIn("1-9")  ~ digits.? )
  def stringChars(c: Char) = c != '\'' && c != '\\'
  def hexDigit[_: P]      = P( CharIn("0-9a-fA-F") )
  def unicodeEscape[_: P] = P( "u" ~ hexDigit ~ hexDigit ~ hexDigit ~ hexDigit )
  def escape[_: P]        = P( "\\" ~ (CharIn("\"/\\\\bfnrt") | unicodeEscape) )
  def alpha[_: P]         = P( CharPred(isLetter) )
  def idRest[_: P]: P[Char] = P( CharPred(c => isLetter(c) | isDigit(c) | c == '_').! ).map(_(0))

  def int[_: P] = P(  CharIn("+\\-").? ~ integral ).!.map(
    x => x.toInt
  )

  def double[_: P] = P(  CharIn("+\\-").? ~ integral ~ fractional ~ exponent.? ).!.map(
    x => x.toDouble
  )

  def number[_: P] = P(  double | int ).map(x => x match {
    case x: Int => x.toDouble
    case x: Double => x
  })

  def strChars[_: P] = P( CharsWhile(stringChars) )

  def string[_: P] =
    P( space ~ "\'" ~/ (strChars | escape).rep.! ~ "\'")

  def letBinding[_: P]: P[TensorLet] = P( "let" ~/ tensorVar ~/ "=" ~/ expr ~/ "in".? ~/ expr).map(x => {
    TensorLet(x._1, x._2, x._3)
  })
  def const[_: P]: P[TensorConst] = P( "tensor" ~/ "(" ~/ int ~/ space ~/ "," ~/ space ~/ intParams ~/ space ~/ "," ~/ space ~/ numParams ~/ space ~/ ")" ).map(x =>
    TensorConst(x._3, TensorDim(x._1, x._2))
  )
  def contract[_: P]: P[TensorContract] = P( "contract" ~/ "(" ~/ string ~/ space ~/ "," ~/ space ~/ (expr).rep(sep=","./) ~/ space ~/ ")" ).map(x =>
    TensorContract(x._2, x._1, EMPTY_DIM)
  )
  def select[_: P]: P[TensorSelect] = P( "select" ~/ "(" ~/ expr ~/ space ~/ "," ~/ space ~/ intParams ~/ space ~/ "," ~/ space ~/ int ~/ space ~/ ")" ).map(x =>
    TensorSelect(x._1, x._2, x._3)
  )
  def normalize[_: P]: P[TensorNormalize] = P( "normalize" ~/ "(" ~/ expr ~/ space ~/ "," ~/ space ~/ intParams ~/ space ~/ ")" ).map(x =>
    TensorNormalize(x._1, x._2) // extend normalize. e.g., normalize(prob_, [0]) := marginalize for the node distance 
  )
  def loadDiscrete[_: P]: P[TensorLoad] = P( "load" ~/ space ~/ "(" ~/ variableString ~/ space ~/ "," ~/ space ~/ ("[" ~/ tensorVar.rep(sep=","./) ~ space ~/ "]") ~/ space ~/ "," ~/ space ~/ intParams ~/ space ~/ ")" ).map(x =>
    TensorLoad(x._1, x._2, TensorDim(x._3.size, x._3.map(z => z.asInstanceOf[Int]).toSeq)))
  def variable[_: P]: P[String] = P( space ~ !keywords ~ ((alpha | "_" | "$") ~ idRest.rep).! ~ space )
  // def variable[_: P]: P[TensorSym] = P( space ~ !keywords ~ ((alpha | "_" | "$") ~ idRest.rep).! ~ space ).map(x => TensorSym(x, EMPTY_DIM))
  def variableString[_: P]: P[String] = P( "'"  ~ variable ~ "'")
  // def discreteParams[_: P] = 
  //   P( ("(" ~/ expr.rep(sep=","./) ~ space ~/ ")") | 
  //       ("[" ~/ expr.rep(sep=","./) ~ space ~/ "]") )
  def ones[_: P]: P[TensorOnes] = P( "ones" ~/ "(" ~/ variableString ~/ space ~/ "," ~/ space ~/ int ~/ space ~/ "," ~/ space ~/ intParams ~/ space ~/ ")" ).map(x =>
    TensorOnes(x._1, TensorDim(x._2, x._3))
  )

  def tensorVar[_: P]: P[TensorSym] = P( variable ).map(x => varNoDim(x)) // FIXME: Doesn't parse the dimension of variable loaded using loadDiscrete correctly
  def intParams[_: P] = 
      P( ("[" ~/ (space ~ int ~ space).rep(sep=","./).? ~ space ~/ "]").map(x => x.getOrElse(Seq())) )
  def numParams[_: P] = 
      P( ("[" ~/ (space ~ number ~ space).rep(sep=","./) ~ space ~/ "]") )
  def factor[_: P]: P[TensorExp] = 
    P(space ~ (const | letBinding | contract | select | normalize | tensorVar | parens | loadDiscrete |
               ones) ~ space)

  def eval(tree: (TensorExp, Seq[(String, TensorExp)])): TensorExp = {
    val (base, ops) = tree
    // println(s"base: $base")
    // println(s"ops: $ops")
    // println(s"ops.foldLeft(base): ${ops.foldLeft(base)}")
    ops.foldLeft(base){ case (left:TensorSym, (op, right:TensorSym)) => op match{
      case "+" => TensorArithmetic(left, right, TensorMathOp.Add)
      case "-" => TensorArithmetic(left, right, TensorMathOp.Subtract)
      case "*" => TensorArithmetic(left, right, TensorMathOp.Multiply)
      case "/" => TensorArithmetic(left, right, TensorMathOp.Divide)
    }}
  }

  def divMul[_: P]: P[TensorExp] = P( factor ~ (StringIn("*", "/").! ~/ factor).rep ).map(eval)
  def addSub[_: P]: P[TensorExp] = P( divMul ~ (StringIn("+", "-").! ~/ divMul).rep ).map(eval)

  // BIL parser
  // def factorMult[_: P]: P[TensorExp] = P( factor ~ (
  //     (".".! ~/ variable) | 
  //     ("^".! ~/ factor) | 
  //     ("(".! ~/ expr.rep(sep=","./) ~/ ")" ~ space)).rep ).map(x => 
  //       x._2.foldLeft(x._1)((acc, cur) => cur match {
  //           case (".", TensorSym(name, _)) => raise("Parsing for factorMult failed: field not supported")
  //           case ("(", e: Seq[_]) => TensorFunCall(acc.asInstanceOf[TensorSym].name, e.asInstanceOf[Seq[TensorExp]])
  //           case ("^", e: TensorExp) => TensorBinOp(acc, e, cur._1)
  //           case _ => raise("Parsing for factorMult failed")
  //         } ))

  // def factorMult[_: P]: P[TensorExp] = P( factor ~ (
  //     ( expr.rep(sep=","./) ~ space )).rep ).map(x => 
  //       x._2.foldLeft(x._1)((acc, cur) => cur match {
  //           // case (".", TensorSym(name, _)) => raise("Parsing for factorMult failed: field not supported")
  //           case (e: Seq[_]) => TensorFunCall(acc.asInstanceOf[TensorSym].name, e.asInstanceOf[Seq[TensorExp]])
  //           // case ("^", e: TensorExp) => TensorBinOp(acc, e, cur._1)
  //           case _ => raise("Parsing for factorMult failed")
  //         } ))
          
  // def divMul[_: P]: P[TensorExp] = P( factorMult ~ (StringIn("*", "/").! ~/ factorMult).rep ).map(x => 
  //   x._2.foldLeft(x._1)((acc, cur) => TensorBinOp(acc, cur._2, cur._1)))

  // def addSub[_: P]: P[TensorExp] = P( divMul ~ (StringIn("+", "-").! ~ !(">") ~/ divMul).rep ).map(x => 
  //   x._2.foldLeft(x._1)((acc, cur) => TensorBinOp(acc, cur._2, cur._1)))

  def expr[_: P]: P[TensorExp] = P( addSub )
  def top[_: P]: P[TensorExp] = P(expr ~ End)

  def parens[_: P]: P[TensorExp] = P( "(" ~/ expr ~/ ")")

  val EMPTY_DIM = TensorDim(0, Seq())

  def varNoDim(name: String) = TensorSym(name, EMPTY_DIM)

  def apply(str: String): TensorExp = {
    val value = parse(str, top(_)) match {
      case Parsed.Success(value, _) => value
      case f @ Parsed.Failure(msg, p, extra) => 
        // println(s"msg: ${msg}")
        // println(s"p: $p")
        // println(s"extra: $extra")
        raise(s"Parse failed `$f` for tensor`$str`")
    }
    // println(s"value: ${value}")
    TensorDimChecker(value)
  }
}
