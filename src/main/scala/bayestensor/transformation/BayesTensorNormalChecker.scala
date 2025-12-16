package bayestensor
package transformation

import ir._
import scala.annotation.tailrec
import utils._

object BayesTensorNormalChecker extends (Exp => Unit) {
  def isValidDist(e: Exp): Boolean = e match {
    case Discrete(_) => true
    // case Variable(_) => true // for assignment to variable in types of int / math exp / ...
    case MatchCase(_, vs) => vs.forall(x => isValidDist(x._2))
    case LoadDiscrete(_,_,_) => true
    case _ => false
  }

  object AcceptableBinding {
    def unapply(e: Exp): Option[Seq[Sym]] = 
      if(isValidDist(e))
        e match {
          case Discrete(_) => Some(Seq())
          case MatchCase(MatchVars(vs), _) => Some(vs)
          case LoadDiscrete(_, vs, _) => Some(vs)
          case _ => ???
        }
      else
        None
  }

  def isAcceptableTensor(e: Exp): Boolean = e match {
    case ToTensor(_) => true
    case LoadTensor(_, _) => true
    case Filter(_,_) => true
    case IntLiteral(_) => true
    case VectorLiteral(_) => true
    case Mult(_,_) => true
    case Sub(_,_) => true
    case Add(_,_) => true
    case Div(_,_) => true
    case _ => false
  }

  object AcceptableTensor {
    def unapply(e: Exp): Option[Exp] = 
      if(isAcceptableTensor(e))
        e match {
          case ToTensor(x) => Some(e)
          case LoadTensor(_, _) => Some(e)
          case Filter(_,_) => Some(e)
          case IntLiteral(x) => Some(e)
          case VectorLiteral(x) => Some(e) 
          case Mult(_,_) => Some(e)
          case Sub(_,_) => Some(e)
          case Add(_,_) => Some(e)
          case Div(_,_) => Some(e)
          case _ => ???
        }
      else
        None
  }

  def isAcceptableNormalization(e: Exp): Boolean = e match {
    case Normalize(_,_) => true
    case _ => false
  }

  object AcceptableNormalization {
    def unapply(e: Exp): Option[(Sym, Seq[Exp])] = 
      if(isAcceptableNormalization(e))
        e match {
          case Normalize(e1,e2) => Some((e1,e2))
          case _ => ???
        }
      else
        None
  }

  def isAcceptableOuterProduct(e: Exp): Boolean = e match {
    case OuterProduct(_,_) => true
    case _ => false
  }

  object AcceptableOuterProduct {
    def unapply(e: Exp): Option[(Sym, Sym)] = 
      if(isAcceptableOuterProduct(e))
        e match {
          case OuterProduct(e1,e2) => Some((e1,e2))
          case _ => ???
        }
      else
        None
  }

  def isAcceptableUnitMatrix(e: Exp): Boolean = e match {
    case UnitMatrix(_) => true
    case _ => false
  }

  object AcceptableUnitMatrix {
    def unapply(e: Exp): Option[Exp] = 
      if(isAcceptableUnitMatrix(e))
        e match {
          case UnitMatrix(x) => Some(e)
          case _ => ???
        }
      else
        None
  }

  def isQueryBinding(e: Exp): Boolean = e match {
    case ObserveNorm(_, _) | SelectNorm(_, _) => true
    case _ => false
  }

  object QueryBinding {
    def unapply(e: Exp): Option[Exp] = 
      if(isQueryBinding(e))
        Some(e)
      else
        None
  }

  def isAcceptableWeights(e: Exp): Boolean = e match {
    case Weight(_,_) => true
    case _ => false
  }

  object AcceptableWeights {
    def unapply(e: Exp): Option[Weight] = 
      if(isAcceptableWeights(e))
        e match {
          case Weight(e1,e2) => Some(Weight(e1,e2))
          case _ => ???
        }
      else
        None
  }

  def isAcceptableDictionary(e: Exp): Boolean = e match {
    case LoadDictionary(_,_) => true
    case _ => false
  }

  object AcceptableDictionary {
    def unapply(e: Exp): Option[Exp] = 
      if(isAcceptableDictionary(e))
        e match {
          case LoadDictionary(_,_) => Some(e)
          case _ => ???
        }
      else
        None
  }

  def isAcceptableReverseMap(e: Exp): Boolean = e match {
    case ReverseMap(_,_) => true
    case _ => false
  }

  object AcceptableReverseMap {
    def unapply(e: Exp): Option[(Exp, Seq[Exp])] = 
      if(isAcceptableReverseMap(e))
        e match {
          case ReverseMap(e, dicts) => Some((e, dicts))
          case _ => ???
        }
      else
        None
  }

  def isAcceptableLift(e: Exp): Boolean = e match {
    case Lift(_) => true
    case _ => false
  }

  object AcceptableLift {
    def unapply(e: Exp): Option[Exp] = 
      if(isAcceptableLift(e))
        e match {
          case Lift(_) => Some(e)
          case _ => ???
        }
      else
        None
  }

  def isAcceptableDecompress(e: Exp): Boolean = e match {
    case Decompress(_,_,_) => true
    case _ => false
  }

  object AcceptableDecompress {
    def unapply(e: Exp): Option[Exp] = 
      if(isAcceptableDecompress(e))
        e match {
          case Decompress(_,_,_) => Some(e)
          case _ => ???
        }
      else
        None
  }

  case class FullQueryCtx(sels: Map[Sym, Select], obss: Map[Sym, Observe], weights: Map[Sym, Weight], dag: Dag[Sym], queries: Seq[(Sym, ProbQuery)])
  object FullQueryCtx {
    def empty: FullQueryCtx = FullQueryCtx(Map(), Map(), Map(), Map(), Seq())
  }

  case class ObsSel(sym: Sym, values: Seq[Int], isObs: Boolean)
  sealed trait ProbQuery
  case class FullQuery(outputVars: Seq[Sym], selections: Seq[Select], observations: Seq[Observe], weights: Seq[Weight], dag: Dag[Sym]) extends ProbQuery {
    def obsSels: Seq[ObsSel] = {
      val sels = selections.map(sel => sel match {
        case SelectNorm(s, is) => ObsSel(s, is, false)
        case _ => raise(s"Select $sel not in normal form")
      })
      val obss = observations.map(obs => obs match {
        case ObserveNorm(s, is) => ObsSel(s, is, true)
        case _ => raise(s"Observe $obs not in normal form")
      })
      sels ++ obss
    }
  }

  case class TensorQuery(e: Exp) extends ProbQuery
  case class DictionaryQuery(e: Exp) extends ProbQuery
  case class ReverseMapQuery(e: Exp, dicts: Seq[Exp]) extends ProbQuery
  case class LiftQuery(e: Exp) extends ProbQuery
  case class DecompressQuery(e: Exp) extends ProbQuery
  case class NormalizeQuery(e: Sym, args: Seq[Exp]) extends ProbQuery
  case class OuterProductQuery(e1: Sym, e2: Sym) extends ProbQuery
  case class UnitMatrixQuery(e: Exp) extends ProbQuery
  case class ExpQuery(normVars: Seq[Sym], dist: Sym, binsMean: Sym, dims: Seq[Exp]) extends ProbQuery
  case class ProgQueries(queries: Seq[ProbQuery], syms: Seq[Sym]) {
    def head = queries.head
    def size = queries.size
  }

  implicit class ObserveOps(o: Observe) {
    def name: String = o match {
      case ObserveNorm(Sym(name), _) => name
      case _ => raise(s"Observe $o not in normal form (random variable)")
    }
    def value: Int = o match {
      case ObserveNorm(_, Seq(i)) => i
      case _ => raise(s"Observe $o not in normal form (params)")
    }
  }

  def singleFullQuery(e: Exp)(ctx: FullQueryCtx): FullQuery = e match {
    case x @ Sym(_) => 
      if(ctx.sels.keySet.nonEmpty)
        raise(s"selection not consistent: ${ctx.sels.keySet} not empty")
      if(ctx.obss.keySet.nonEmpty)
        raise(s"observe not consistent: ${ctx.obss.keySet} not empty")
      FullQuery(Seq(x), Seq(), Seq(), Seq(), ctx.dag)
    case Query(seq, sels, obss, weights) => //if seq.size > 0 =>
      // println(s"ctx.sels.keySet: ${ctx.sels.keySet}")
      // println(s"sels.toSet: ${sels.toSet}") 
      if(ctx.sels.keySet != sels.toSet)
        raise(s"selection not consistent: ${ctx.sels.keySet} != ${sels.toSet}")
      if(ctx.obss.keySet != obss.toSet)
        raise(s"observe not consistent: ${ctx.obss.keySet} != ${obss.toSet}")
      // println(s"after sels.map(ctx.sels): ${sels.map(ctx.sels)}")
      // println(s"after ctx.sels.keySet: ${ctx.sels}")
      // println(s"weights: ${weights}")
      // println(s"ctx.weights: ${ctx.weights}")
      FullQuery(seq, sels.map(ctx.sels), obss.map(ctx.obss), weights.map(ctx.weights), ctx.dag)
    case _ => ???  
  }
  def singleExpectation(ex: Expectation): ExpQuery = ex match {
    case Expectation(normVars, dist: Sym, binsMean: Sym, dims) =>
      ExpQuery(normVars, dist, binsMean, dims)
    case _ =>
      raise(s"Expectation `$ex` is not in the required format.")
  }

  @tailrec
  def fullQuery(e: Exp)(ctx: FullQueryCtx): ProgQueries = e match {
    case LetBinding(x, AcceptableBinding(ps), e2) => fullQuery(e2)(ctx.copy(dag = ctx.dag.addEdges(x, ps)))
    case LetBinding(x, AcceptableTensor(ps), e2) => fullQuery(e2)(ctx.copy(queries = ctx.queries :+ (x -> TensorQuery(ps))))
    case LetBinding(x, AcceptableUnitMatrix(ps), e2) => fullQuery(e2)(ctx.copy(queries = ctx.queries :+ (x -> UnitMatrixQuery(ps))))
    case LetBinding(x, AcceptableOuterProduct(ps1, ps2), e2) => fullQuery(e2)(ctx.copy(queries = ctx.queries :+ (x -> OuterProductQuery(ps1, ps2))))
    case LetBinding(x, AcceptableNormalization(ps1, ps2), e2) => fullQuery(e2)(ctx.copy(queries = ctx.queries :+ (x -> NormalizeQuery(ps1, ps2))))
    case LetBinding(x, o @ ObserveNorm(_, _), e2) => fullQuery(e2)(ctx.copy(obss = ctx.obss + (x -> o)))
    case LetBinding(x, s @ SelectNorm(_, _), e2)  => fullQuery(e2)(ctx.copy(sels = ctx.sels + (x -> s)))
    case LetBinding(x, AcceptableWeights(ps), e2) => fullQuery(e2)(ctx.copy(weights = ctx.weights + (x -> ps)))
    case LetBinding(x, AcceptableDictionary(ps), e2) => fullQuery(e2)(ctx.copy(queries = ctx.queries :+ (x -> DictionaryQuery(ps))))
    case LetBinding(x, AcceptableReverseMap(ps1, ps2), e2) => fullQuery(e2)(ctx.copy(queries = ctx.queries :+ (x -> ReverseMapQuery(ps1, ps2))))
    case LetBinding(x, AcceptableLift(ps), e2) => fullQuery(e2)(ctx.copy(queries = ctx.queries :+ (x -> LiftQuery(ps))))
    case LetBinding(x, AcceptableDecompress(ps), e2) => fullQuery(e2)(ctx.copy(queries = ctx.queries :+ (x -> DecompressQuery(ps))))
    case LetBinding(x, q @ Query(_, _, _, _), e2) => 
      val fq1 = singleFullQuery(q)(ctx)
      fullQuery(e2)(ctx.copy(sels = Map(), obss = Map(), weights = Map(), dag = ctx.dag.addEdges(x, Seq()), queries = ctx.queries :+ (x -> fq1)))
    case LetBinding(x, ex @ Expectation(_, _, _, _), e2) => 
      val fq1 = singleExpectation(ex)
      fullQuery(e2)(ctx.copy(queries = ctx.queries :+ (x -> fq1)))
    case LetBinding(x, e1, e2) =>
      raise(s"Let-binding rhs not in normal form: $e1")
    case Sym(_) | Query(_, _, _, _) | ToTensor(_) | 
         Expectation(_, _, _, _) | Normalize(_, _) | 
         OuterProduct(_, _) => 
      val (ss, qs) = ctx.queries.unzip
      if(e.isInstanceOf[Sym] && qs.size > 0)
        ProgQueries(qs, ss)
      else 
        ProgQueries(qs :+ singleFullQuery(e)(ctx), ss :+ Sym("__res"))
    case _ => raise(s"Normal form doesn't support:`$e`")
  }

  def check(e: Exp): Unit = fullQuery(e)(FullQueryCtx.empty)
  def apply(e: Exp): Unit = check(e)
}