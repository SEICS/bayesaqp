package bayestensor
package transformation

import ir._
import scala.annotation.tailrec
import utils._

object BayesTensorNormalizer extends (Exp => Exp) {
  def normalize(e: Exp): Exp = {
    e |> 
      // debug("before normalize") |>
      anf |> 
      // debug("after ANF") |>
      varExpand(Map(), false) |> 
      // debug("after varExpand") |>
      matchMerge(Map(), false) |> 
      // debug("after matchMerge") |>
      queryIntroduce(Seq(), Seq(), Seq())
  }
  def debug(msg: String)(e: Exp): Exp = {
    println(s"**BEGIN $msg**")
    println(backend.BayesTensorCodeGenerator(e))
    println(s"**END $msg**")
    e
  }
  def anf(e: Exp): Exp = {
    val (res, seq) = bindings(e, false)
    LetBindingN(seq, res)
  }
  def isComplex(e: Exp): Boolean = e match {
    case Sym(_) | IntLiteral(_) | DoubleLiteral(_) => false
    case _ => true
  }
  def bindings(e: Exp, letRhs: Boolean): (Exp, Seq[(Sym, Exp)]) = e match {
    case LetBinding(x, e1, e2) if !letRhs =>
      val (re1, se1) = bindings(e1, true)
      val (re2, se2) = bindings(e2, false)
      re2 -> (se1 ++ Seq(x -> re1) ++ se2)
    case IfThenElse(_, _, _) =>
      raise("if-then-else shouldn't exist!")
    case MatchCase(vs @ MatchVars(_), cs) =>
      val rscs = cs.map(x => bindings(x._2, true))
      if(rscs.exists(_._2.nonEmpty))
        raise(s"not in normal form $e")
      if(letRhs)
        e -> Seq()
      else {
        val x = Sym.fresh()
        x -> Seq(x -> e)
      }
    case Observe(e, vs) =>
      val (te, se) = bindings(e, false)
      if(letRhs)
        Observe(te, vs) -> se
      else {
        val x = Sym.fresh()
        x -> (se ++ Seq(x -> Observe(te, vs)))
      }
    case ExpShape(cs, fact) =>
      val (tcs, scs) = cs.map(x => bindings(x, false)).unzip
      fact(tcs) -> scs.flatten
  }
  def queryIntroduce(sels: Seq[(Sym, Select)], obss: Seq[(Sym, Observe)], weights: Seq[(Sym, Weight)])(e: Exp): Exp = e match {
    case LetBinding(x, e1 @ Observe(_, _), e2) => 
      LetBinding(x, e1, queryIntroduce(sels, obss :+ (x -> e1), weights)(e2))
    case LetBinding(x, e1 @ Select(_, _), e2) => 
      LetBinding(x, e1, queryIntroduce(sels :+ (x -> e1), obss, weights)(e2))
    case LetBinding(x, e1, e2) =>
      LetBinding(x, e1, queryIntroduce(sels, obss, weights)(e2))
    case s @ Sym(_) => 
      Query(Seq(s), sels.map(_._1), obss.map(_._1), weights.map(_._1)) // ?
    // add expectation here
    case _ =>
      raise(s"queryIntroduce doesn't support $e")
  }
  case class MatchCasePath(vars: Seq[Sym], bindings: Map[Sym, Int], body: Discrete) {
    def addBinding(binding: (Sym, Int)): MatchCasePath = 
      MatchCasePath(binding._1 +: vars, bindings + binding, body)
  }
  def getVars(e: Exp): Seq[Sym] = e match {
    case MatchCase(e1, cs) => getVars(e1) ++ cs.flatMap(x => getVars(x._2))
    case s @ Sym(_) => Seq(s)
    case Discrete(_) | IntLiteral(_) | DoubleLiteral(_) => Seq()
    case _ => raise(s"getVars doesn't support $e")
  }
  def getPaths(e: Exp): Seq[MatchCasePath] = e match {
    case MatchCase(MatchVars(Seq(s)), cs) =>
      cs.flatMap(c => c match {
        case (IntLiteral(i), body) => 
          getPaths(body).map(path =>
            path.addBinding(s -> i)
          )
        case _ => raise(s"MathCase needs literal, but found `${c._1}`")
      })
    case d @ Discrete(_) => Seq(MatchCasePath(Seq(), Map(), d))
    case _ => raise(s"getPaths doesn't support $e")
  }
  def matchMerge(ctx: Map[Sym, Exp], inMatch: Boolean)(e: Exp): Exp = e match {
    case LetBinding(x, e1, e2) =>
      // println(s"matchMerge for variable $x's rhs")
      val te1 = matchMerge(ctx, inMatch)(e1)
      // println(s"matchMerge for variable $x's body")
      val te2 = matchMerge(ctx + (x -> te1), inMatch)(e2)
      LetBinding(x, te1, te2)
    case MatchCase(e1 @ MatchVars(inVars), cs) =>
      val vs = cs.map(x => getVars(x._2))
      val uvs = (inVars ++ vs.flatten).distinct
      val paths = getPaths(e)
      val uvsDim = uvs.map(v => v -> getDim(ctx(v))).toMap
      def cartProd(sizes: Seq[Int]): Seq[Seq[Int]] = 
        sizes.tail.foldLeft[Seq[Seq[Int]]]((0 until sizes.head).map(i => Seq(i)).toSeq)((acc, cur) =>
          acc.flatMap(a =>
            (0 until cur).map(i => a :+ i)
          )
        )

      val idxs = cartProd(uvs.map(uvsDim))
      def normalizePath(p: MatchCasePath): Seq[MatchCasePath] = {
        val rvs = uvs.filter(v => !p.vars.contains(v))
        rvs.foldLeft(Seq(p))((acc, v) => {
          val n = uvsDim(v)
          acc.flatMap(p1 =>
            (0 until n).map(i => p1.addBinding(v -> i))
          )
        })
      }
      val normalizedPaths = paths.flatMap(normalizePath)
      // println(s"idxs size: ${idxs.size}")
      // println(s"normalizedPaths size: ${normalizedPaths.size}")
      // println(s"paths size: ${paths.size}")
      // if(normalizedPaths.size != idxs.size)
      //   println(s"paths: ${paths.mkString("\n\t")}")
      assert(normalizedPaths.size == idxs.size)
      // val pathsMap = paths.map(p => p.bindings -> p).toMap
      val pathsMap = normalizedPaths.map(p => p.bindings -> p).toMap
      def longestPath(idx: Seq[Int]): MatchCasePath = {
        val bind = uvs.zip(idx).toMap
        pathsMap.get(bind) match {
          case Some(p) => p
          case None => 
            // longestPath(idx.dropRight(1))
            raise(s"path with index `$idx` not found")
        }
      }
      val ncs = idxs.map(idx => TupleN(idx.map(IntLiteral)) -> longestPath(idx).body)
      // println(ncs.mkString("\n"))
      // println(paths)
      MatchCase(MatchVars(uvs), ncs)
    case Discrete(_) | IntLiteral(_) | 
         DoubleLiteral(_) | Observe(_, _) |
         Select(_, _) | Sym(_) =>
      e
    case _ => raise(s"matchMerge doesn't handle `$e`")
  }
  def getDim(e: Exp): Int = e match {
    case LetBinding(x, e1, e2) => raise(s"Doesn't know how to compute dim for let-binding $e")
    case Discrete(vs) => vs.length
    case MatchCase(_, vs) => getDim(vs.head._2)
    case Observe(_, _) | Select(_, _) => -1
    case _ => raise(s"Doesn't know how to compute dim for $e")
  }
  def oneHot(i: Int, n: Int) = (0 until n).map(j => if(i == j) 1.0 else 0.0).map(DoubleLiteral)
  def varExpand(ctx: Map[Sym, Int], inMatch: Boolean)(e: Exp): Exp = e match {
    case s @ Sym(_) if inMatch => 
      val n = ctx(s)
      MatchCase(s, (0 until n).map(i => IntLiteral(i) -> Discrete(oneHot(i, n))))
    case s @ Sym(_) => s
    case LetBinding(x, e1, e2) =>
      val te1 = varExpand(ctx, inMatch)(e1)
      val n = getDim(te1)
      val te2 = varExpand(ctx + (x -> n), inMatch)(e2)
      LetBinding(x, te1, te2)
    case MatchCase(e1, cs) =>
      val te1 = varExpand(ctx, false)(e1)
      val tcs = cs.map(x => x._1 -> varExpand(ctx, true)(x._2))
      MatchCase(te1, tcs)
    case IntLiteral(i) if inMatch => // FIXME
      Discrete(oneHot(i, 2))
    case Discrete(_) | IntLiteral(_) | 
         DoubleLiteral(_) | Observe(_, _) |
         Select(_, _) =>
      e
    case _ => raise(s"varExpand doesn't handle `$e`")
  }
  def apply(e: Exp): Exp = normalize(e)
}