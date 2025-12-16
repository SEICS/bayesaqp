package bayestensor
package frontend

import ir._
import fastparse._, NoWhitespace._, CharPredicates._


object BayesTensorParser {
  def keywords[_: P] = P (
    StringIn("if", "then", "else", "let", "false", 
      "observe", "discrete", "match", "with", "flip",
      "fst", "snd", "select", "query", "fun",
      "true", "in", "int", "double", "expectation", 
      "vector", "tensor", "normalize", "outer", "loadDiscrete",
      "unitMat", "loadTensor", "filter", "weight", "loadDictionary",
      "lift", "map", "decompress") ~
      !idRest
  )  
  // extends keyword set to include exp at here
  def `false`[_: P]       = P( "false" ).map(_ => BoolLiteral(false))
  def `true`[_: P]        = P( "true" ).map(_ => BoolLiteral(true))

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
  def stringChars(c: Char) = c != '\"' && c != '\\'
  def hexDigit[_: P]      = P( CharIn("0-9a-fA-F") )
  def unicodeEscape[_: P] = P( "u" ~ hexDigit ~ hexDigit ~ hexDigit ~ hexDigit )
  def escape[_: P]        = P( "\\" ~ (CharIn("\"/\\\\bfnrt") | unicodeEscape) )
  def alpha[_: P]         = P( CharPred(isLetter) )
  def idRest[_: P]: P[Char] = P( CharPred(c => isLetter(c) | isDigit(c) | c == '_').! ).map(_(0))
  
  def int[_: P] = P(  CharIn("+\\-").? ~ integral ).!.map(
    x => IntLiteral(x.toInt)
  )

  def boundedInt[_: P] = P(  "int" ~/ space ~/ "(" ~/ space ~/ int ~/ space ~/ "," ~/ space ~/ int ~/ space ~/ ")" ).map(
    x => x._2
  )

  def number[_: P] = P(  CharIn("+\\-").? ~ integral ~ fractional ~ exponent.? ).!.map(
    x => DoubleLiteral(x.toDouble)
  )
  def ifThenElse[_: P]: P[IfThenElse] = P( "if" ~/ expr ~/ "then" ~/ expr ~/ "else" ~/ expr).map(x => IfThenElse(x._1, x._2, x._3))
  def letBinding[_: P]: P[LetBinding] = P( "let" ~/ variable ~/ "=" ~/ expr ~/ "in".? ~/ expr).map(x => LetBinding(x._1, x._2, x._3))
  def const[_: P]: P[Const] = P(`true` | `false` | number | int | boundedInt)
  def variable[_: P]: P[Sym] = P( space ~ !keywords ~ ((alpha | "_" | "$") ~ idRest.rep).! ~ space ).map(x => Sym(x))
  def variableString[_: P]: P[String] = P( "'"  ~ variable ~ "'").map(x => x.name)
  def observe[_: P]: P[Observe] = P( "observe" ~/ expr ~/ ("[" ~/ expr.rep(sep=","./) ~ space ~/ "]").?).map(x => Observe(x._1, x._2.getOrElse(Seq())))
  def select[_: P]: P[Select] = P( "select" ~/ expr ~/ "[" ~/ expr.rep(sep=","./) ~ space ~/ "]").map(x => Select(x._1, x._2))
  def query[_: P]: P[Query] = P( "query" ~/ space ~/
    "[" ~/ variable.rep(sep=","./) ~/ space ~/ "]" ~/ space ~/
    "[" ~/ variable.rep(sep=","./) ~/ space ~/ "]" ~/ space ~/
    "[" ~/ variable.rep(sep=","./) ~/ space ~/ "]" ~/ space ~/
    "[" ~/ variable.rep(sep=","./) ~/ space ~/ "]").map(x => Query(x._1, x._2, x._3, x._4))
  def flip[_: P]: P[Flip] = P( "flip" ~/ expr ).map(x => Flip(x))
  def discreteParams[_: P] = 
      P( ("(" ~/ expr.rep(sep=","./) ~ space ~/ ")") | 
         ("[" ~/ expr.rep(sep=","./) ~ space ~/ "]") )
  def vectorParams[_: P] = 
      P( "[" ~/ expr.rep(sep=","./).map(x => x.map(y => y match {
        case DoubleLiteral(d) => d
        case IntLiteral(i) => i.toDouble
        case _ => raise(s"Parsing for vector failed: number expected but `$y` provided")
      })) ~ space ~/ "]" )
  def discrete[_: P]: P[Discrete] = P( "discrete" ~/ space ~/ discreteParams ).map(x => Discrete(x))
  def loadDiscrete[_: P]: P[LoadDiscrete] = P( "loadDiscrete" ~/ space ~/ "(" ~/ variableString ~/ space ~/ "," ~/ space ~/ ("[" ~/ variable.rep(sep=","./) ~ space ~/ "]") ~/ space ~/ "," ~/ space ~/ discreteParams ~/ space ~/ ")" ).map(x => LoadDiscrete(x._1, x._2, x._3))
  def loadTensor[_: P]: P[LoadTensor] = P( "loadTensor" ~/ space ~/ "(" ~/ variableString ~/ space ~/ "," ~/ space ~/ discreteParams ~/ space ~/ ")" ).map(x => LoadTensor(x._1, x._2))
  def loadDictionary[_: P]: P[LoadDictionary] = P( "loadDictionary" ~/ space ~/ "(" ~/ variableString ~/ space ~/ "," ~/ space ~/ discreteParams ~/ space ~/ ")" ).map(x => LoadDictionary(x._1, x._2))
  def lift[_:P]: P[Lift] = P( "lift" ~/ space ~/ expr).map(x => Lift(x))
  def map[_:P]: P[ReverseMap] = P( "map" ~/ space ~/ expr ~/ space ~/ discreteParams).map(x => ReverseMap(x._1, x._2))
  def filter[_: P]: P[Filter] = P( "filter" ~/ expr ~/ "[" ~/ expr.rep(sep=","./) ~ space ~/ "]").map(x => Filter(x._1, x._2))
  def decompress[_:P]: P[Decompress] = P( "decompress" ~/ space ~/ expr ~/ space ~/ expr ~/ space ~/ discreteParams).map(x => Decompress(x._1, x._2, x._3))

  def vectorLiteral[_: P]: P[VectorLiteral] = P( "vector" ~/ space ~/ vectorParams ).map(x => VectorLiteral(x))
  def toTensor[_: P]: P[ToTensor] = P( "tensor" ~/ space ~/ variable ).map(x => ToTensor(x))
  def unitMatrix[_: P]: P[UnitMatrix] = P( "unitMat" ~/ space ~/ variable ).map(x => UnitMatrix(x))
  def matchCase[_: P]: P[MatchCase] =
    P( "match" ~/ expr ~/ "with" ~/ space ~/ "{" ~/ (expr ~/ "->" ~/ expr).rep(sep="|"./) ~ space ~/ "}").map(x => MatchCase(x._1, x._2))
  def fst[_: P]: P[Fst] = P( "fst" ~/ expr ).map(x => Fst(x))
  def snd[_: P]: P[Snd] = P( "snd" ~/ expr ).map(x => Snd(x))
  // def funCall[_: P]: P[Exp] = P( variable ~/ "(" ~/ expr.rep(sep=","./) ~ space ~/ ")").map(x => FunCall(x._1.name, x._2))

  // Add the format for parsing expectation below
  def expectation[_:P]: P[Expectation] = P( "expectation" ~/ space ~/
    "[" ~/ variable.rep(sep=","./) ~/ space ~/ "]" ~/ space ~/
    variable ~/ space ~/
    variable ~/ space ~/ "[" ~/ expr.rep(sep=","./) ~ space ~/ "]").map(x => Expectation(x._1, x._2, x._3, x._4))

  def normalize[_: P]: P[Normalize] = P("normalize" ~/ space ~/
  variable ~/ space ~/ ("[" ~/ expr.rep(sep=","./) ~ space ~/ "]").?).map(x => Normalize(x._1, x._2.getOrElse(Seq())))
  // def normalize[_: P]: P[Normalize] = P( "normalize" ~/ expr ~/ ("[" ~/ expr.rep(sep=","./) ~ space ~/ "]").?).map(x => Normalize(x._1, x._2.getOrElse(Seq())))

  def outerProduct[_: P]: P[OuterProduct] = P("outer" ~/ space ~/
  variable ~/ space ~/ variable).map(x => OuterProduct(x._1, x._2))

  def weight[_: P]: P[Weight] = P("weight" ~/ space ~/
  variable ~/ space ~/ variable).map(x => Weight(x._1, x._2))

  def factor[_: P]: P[Exp] = P(space ~ (const | neg | not | query |
    observe | discrete | flip | matchCase | fst | snd | select | 
    vectorLiteral | toTensor | normalize | outerProduct |
    ifThenElse | letBinding | variable | parensPair | expectation |
    loadDiscrete | unitMatrix | loadTensor | filter | weight |
    loadDictionary | lift | map | decompress) ~ space)

  def neg[_: P]: P[Exp] = P( "-" ~ !(">") ~ factor ).map(x => UnaryOp(x, "-"))
  def not[_: P]: P[Exp] = P( "!" ~ factor ).map(x => UnaryOp(x, "!"))
  def factorMult[_: P]: P[Exp] = P( factor ~ (
      (".".! ~/ variable) | 
      ("^".! ~/ factor) | 
      ("(".! ~/ expr.rep(sep=","./) ~/ ")" ~ space)).rep ).map(x => 
        x._2.foldLeft(x._1)((acc, cur) => cur match {
            case (".", Sym(name)) => raise("Parsing for factorMult failed: field not supported")
            case ("(", e: Seq[_]) => FunCall(acc.asInstanceOf[Sym].name, e.asInstanceOf[Seq[Exp]])
            case ("^", e: Exp) => BinOp(acc, e, cur._1)
            case _ => raise("Parsing for factorMult failed")
          } ))
  def divMul[_: P]: P[Exp] = P( factorMult ~ (StringIn("*", "/", "&&", "||", "~>").! ~/ factorMult).rep ).map(x => 
    x._2.foldLeft(x._1)((acc, cur) => BinOp(acc, cur._2, cur._1)))
  def addSub[_: P]: P[Exp] = P( divMul ~ (StringIn("+", "-").! ~ !(">") ~/ divMul).rep ).map(x => 
    x._2.foldLeft(x._1)((acc, cur) => BinOp(acc, cur._2, cur._1)))
  def addSubCmp[_: P]: P[Exp] = P( addSub ~ (StringIn("<", "==", "<=", "!=", ">", ">=").! ~/ addSub).? ).map(x => 
      x._2 match {
        case Some((op, y)) => BinOp(x._1, y, op)
        case None => x._1
      }
  )
  def parensPair[_: P]: P[Exp] = P( "(" ~/ expr ~/ ("," ~/ expr).? ~/ ")").map(x => x._2 match {
    case Some(e2) => Pair(x._1, e2)
    case None => x._1
  })
  def expr[_: P]: P[Exp] = P( addSubCmp )
  def fun[_: P]: P[Func] = P("fun" ~/ space ~/ variable ~/ space ~/ "(" ~/ space ~/ 
                              (variable ~/ ":" ~/ variable).rep(sep=","./) ~/ space ~/ ")" ~/ 
                              space ~/ "{" ~/ expr ~/ "}").map(x => Func(x._1.name, x._2.map(_._1), x._3))
  def prog[_: P]: P[Prog] = P((fun ~/ space).rep() ~/ space ~/ expr).map(x => Prog(x._1, x._2))
  def top[_: P]: P[Prog] = P(prog ~ End)

  def apply(str: String): Prog = {
    val value = parse(str, top(_)) match {
      case Parsed.Success(value, _) => value
      case f @ Parsed.Failure(msg, p, extra) => raise(s"Parse failed `$f` for bt`$str`")
    }
    value
  }
}
