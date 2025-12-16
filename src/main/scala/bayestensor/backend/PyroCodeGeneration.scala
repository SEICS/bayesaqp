// package bayestensor
// package backend

// import scala.annotation.tailrec
// import transformation._
// import ir._
// import utils._

// object PyroCodeGenerator {
//   import BayesTensorNormalChecker._
//   case class PyroCode(code: Document, map: Map[Sym, TensorDim])
//   def codegen(e: Exp): String = {
//     val queries = fullQuery(e)(FullQueryCtx.empty)
//     assert(queries.size == 1)
//     val query = queries.head
//     val syms = collectSymbols(e)(Seq()).map(_.name)
//     val lastSymbol = query.outputVars.head
//     val model = rcr(e)(Map())(query)
//     val code = doc"""
// @pyro.infer.config_enumerate
// def model(${lastSymbol.name} = None): #{  # ${model.code}  #  #} 

// p_conditioned = pyro.condition(model, data={${query.observations.map(o => doc"'${o.name}': torch.tensor(${o.value})").mkDocument(", ")}})
// p_conditioned_enum = pyro.poutine.enum(p_conditioned, first_available_dim=-1)
// for i in range(0, ${model.map(lastSymbol).size}):
//   trace = pyro.poutine.trace(p_conditioned_enum).get_trace(${lastSymbol.name} = torch.tensor(i))
//   log_prob_evaluate = pyro.infer.mcmc.util.TraceEinsumEvaluator(
//       trace, has_enumerable_sites=True, max_plate_nesting=0)
//   print(f"p(${lastSymbol.name}={i}):", log_prob_evaluate.log_prob(trace).exp())
//     """
//     code.toString
//   }
//   @tailrec
//   def collectSymbols(e: Exp)(res: Seq[Sym]): Seq[Sym] = e match {
//     case LetBinding(x, e1 @ AcceptableBinding(_), e2) =>
//       collectSymbols(e2)(res :+ x)
//     case LetBinding(x, e1, e2) if isQueryBinding(e1) =>
//       collectSymbols(e2)(res)
//     case sym @ Sym(_) =>
//       res
//     case Query(_, _, _) =>
//       res
//     case _ =>
//       raise(s"collectSymbols not supported for $e")
//   }
//   // @tailrec
//   // def returnSymbol(e: Exp): Sym = e match {
//   //   case LetBinding(x, e1, e2) => returnSymbol(e2)
//   //   case x @ Sym(_) => x
//   //   case Query(Seq(x), _, _) => x
//   //   case _ => raise(s"returnSymbol not found for $e")
//   // }
//   def cptName(name: String): String = s"CPT_${name}"
//   def codegenCPT(e: TensorConst): Document = 
//     doc"torch.tensor([${e.elems.mkString(", ")}]).reshape((${e.dim.dims.mkString(", ")}))"
//   def rcr(e: Exp)(ctx: Map[Sym, TensorDim])(implicit query: FullQuery): PyroCode = e match {
//     case LetBinding(x, e1, e2) =>
//       e1 match {
//         case AcceptableBinding(_) =>
//           val cpt = CPT.fromExp(e1)(ctx)
//           val cx = cptName(x.name)
//           val PyroCode(body, newCtx) = rcr(e2)(ctx + (x -> cpt.dim))
//           val dist = e1 match {
//             case Discrete(_) => doc"$cx"
//             case MatchCase(Sym(s1), _) => doc"$cx[$s1]"
//             case MatchCase(Pair(Sym(s1), Sym(s2)), _) => doc"$cx[$s1, $s2]"
//             case _ => ???
//           }
//           PyroCode(doc"""$cx = ${codegenCPT(cpt)}
// ${x.name} = pyro.sample('${x.name}', dist.Categorical($dist)${if(query.outputVars.contains(x)) doc", obs=${x.name}" else doc""}, infer={"enumerate": "parallel"})
// $body""", newCtx)
//         case _ if isQueryBinding(e1) =>
//           rcr(e2)(ctx)
//         case _ => raise(s"Let-binding to $e1 not supported")
//       }
//     case Sym(name) =>
//       PyroCode(doc"$name", ctx)
//     case Query(Seq(Sym(name)), _, _) =>
//       PyroCode(doc"$name", ctx)
//     case _ =>
//       raise(s"Pyro code generation for $e not supported!")
//   }
// }
