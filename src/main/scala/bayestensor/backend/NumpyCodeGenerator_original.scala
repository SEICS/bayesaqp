// package bayestensor
// package backend

// import ir._
// import utils._

// object NumpyCodeGenerator extends (TensorExp => String) {
//   val OPT_EINSUM = true
//   def apply(e: TensorExp): String = 
//     s"""|t_start = perf_counter()
//         |with shared_intermediates():
//         |${codegen(e)}
//         |t_end = perf_counter()
//         |latency = (t_end-t_start)*1000
//         |latency,${codegenRes(e)}""".stripMargin

//     // // q7: [list(chain.from_iterable(customer_c_city_b2v.flat[0].values())), list(chain.from_iterable(supplier_s_city_b2v.flat[0].values())), list(chain.from_iterable(dwdate_d_year_b2v.flat[0].values()))]
//     // // q11: groupby_map = [list(chain.from_iterable(dwdate_d_year_b2v.flat[0].values())), list(chain.from_iterable(supplier_s_city_b2v.flat[0].values())), list(chain.from_iterable(part_p_brand1_b2v.flat[0].values()))]
//     // s"""|t_start = perf_counter()
//     //     |${codegen(e)}
//     //     |groupby_map = [list(chain.from_iterable(customer_c_city_b2v.flat[0].values())), list(chain.from_iterable(supplier_s_city_b2v.flat[0].values())), list(chain.from_iterable(dwdate_d_year_b2v.flat[0].values()))]
//     //     |final_aqp_res = []
//     //     |for idx_matching in list(zip(*np.nonzero(${codegenRes(e)}), ${codegenRes(e)}[np.nonzero(${codegenRes(e)})])):
//     //     |\tassert len(groupby_map) == len(idx_matching)-1
//     //     |\tval = idx_matching[-1]
//     //     |\tnew_aqp_record = []
//     //     |\tfor (a, attr) in enumerate(idx_matching[:-1]):
//     //     |\t\tnew_aqp_record.append(groupby_map[a][attr])
//     //     |\tfinal_aqp_res.append(tuple(new_aqp_record+[val]))
//     //     |${codegenRes(e)}= final_aqp_res
//     //     |t_end = perf_counter()
//     //     |latency = (t_end-t_start)*1000
//     //     |latency,${codegenRes(e)}""".stripMargin

//   def codegenRes(e: TensorExp): String = e match {
//     case TensorLetN(_, TensorSym(name, dim)) => s"${name}_"
//       // s"np.array(list(zip(*np.nonzero(${name}_), ${name}_[np.nonzero(${name}_)])))" // slower
//     // case TensorLetN(_, TensorSym(name, dim)) => s"${name}_"
//       // if (dim.order > 0){
//       //   s"${name}_.tolist()"
//       // } else {
//       //   s"${name}_" //s"print('result:', ${name}_)"
//       // }
//     case _ => ""
//   }
//   def codegen(e: TensorExp): String = 
//     // s"""|start_t = perf_counter()
//     //     |${rcr(e).toString}
//     //     |inference_time = (perf_counter() - start_t) * 1000
//     //     |print(f'inference_time: {inference_time}')
//     // """
//     rcr(e).toString
//     // s"""|cpds = np.load('~/Desktop/bayestensor/progs/aqp/flights/cpds.npz')
//     //     |${rcr(e)}""".stripMargin
//   def rcr(e: TensorExp): Document = e match {
//     case TensorConst(vs, TensorDim(n, ds)) => 
//       val arr = doc"np.array([${vs.mkString(", ")}])"
//       if(n == 0)
//         vs(0).toString
//       else if(n == 1)
//         arr
//       else
//         // doc"$arr"
//         doc"$arr.reshape((${ds.mkString(",")}))"
//     case TensorSelect(e, ids, filterDim) =>
//       filterDim match {
//         // case 0 => doc"${rcr(e)}[[${ids.mkString(",")}]]" // same efficiency as below
//         case _ =>
//           // doc"${rcr(e)}[${":,"* (filterDim - 1)} [${ids.mkString(",")}]]" // same efficiency as below
//           doc"np.take(${rcr(e)}, [${ids.mkString(",")}], axis=${filterDim})"
//       }
//     case TensorNormalize(e, normDims) =>
//       val inputVars = e.dim.dims.zipWithIndex.map(vi => vi._2 -> ('a' + vi._2).toChar).toMap.values.mkString("")
//       val outputVars = if (!normDims.isEmpty){ 
//         normDims.map(i => inputVars(i)).toSeq.mkString("")
//       } else {""}
//       val einsum = s"${inputVars}->${outputVars}" // "dy->y"
//       val finalDim = TensorDim(outputVars.length, normDims.map(dimX => e.dim.dims(dimX)).toSeq)
//       val normJoint = TensorContract(Seq(e), einsum, finalDim)
//       doc"${rcr(e)} / ${rcr(normJoint)}" // shape to divide must match
//     case TensorLet(x, e1, e2) =>
//       doc"${x.name}_ = ${rcr(e1)} # ${rcr(e2)}"
//     case TensorLoad(name, _, dim) =>
//       doc"${name}.reshape((${dim.dims.mkString(",")}))"
//     case TensorSym(name, _) =>
//       doc"${name}_"
//     case DictionaryLoad(name, _) => 
//       doc"${name}.flat[0]"
//     case TensorContract(args, einsum, _) =>
//       val method = 
//         if(OPT_EINSUM) 
//           doc"contract" // contract_expression can be cached with shared_intermediates():
//         else
//           doc"np.einsum"
//       val opt_flag = 
//         if (OPT_EINSUM)
//           doc", optimize='dp', use_blas=True" // , order='C' flag 'dp' faster for Flights but slow for SSB // flag 'greedy' faster for higher-order tensor algebra (e.g., SSB's q11 (indexed from 0) will be 10x faster using 'greedy' than 'dp')
//         else doc""
//       doc"$method('$einsum', ${args.map(rcr).mkDocument(", ")}$opt_flag)"
//     case TensorArithmetic(e1, e2, op) =>
//       // println(s"rcr(e1): ${op}")
//       if (!"+*-/".contains(op.opStr)) {
//         raise(s"Doesn't support math operator `$op`.")
//       } 
//       if (op.opStr == "/") { // to avoid producing Nans because of number/0
//         doc"np.divide(${rcr(e1)}, ${rcr(e2)}, out=np.zeros_like(${rcr(e1)}), where=${rcr(e2)}!=0)"
//       } else {
//         doc"${rcr(e1)} ${op.opStr} ${rcr(e2)}"
//       }
//     case TensorOnes(name, dim) =>
//       doc"np.ones((${dim.dims.mkString(",")}))"
//     case TensorReverseMap(e, dicts, _) => 
//       val dictList = dicts.map(d => s"${d.asInstanceOf[TensorSym].name}_").toSeq
//       doc"reverse_map(${rcr(e)}, ${dictList.mkString(", ")})"
//     case TensorLift(e) => 
//       doc"np.array([k + (v,) for (k,v) in ${rcr(e)}.items()])" // ${rcr(e)}.reshape((${e.dim.dims.mkString(",")}))
//     case _ => ???
//   }
// }
