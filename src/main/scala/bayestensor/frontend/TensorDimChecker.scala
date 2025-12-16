package bayestensor
package frontend

import ir._

object TensorDimChecker extends (TensorExp => TensorExp) {
  def apply(exp: TensorExp): TensorExp = 
    dimCheck(exp)(Map())
  def dimCheck(exp: TensorExp)(ctx: Map[String, TensorDim]): TensorExp = exp match {
    case TensorLet(x, e1, e2) =>
      val te1 = dimCheck(e1)(ctx)
      val tx = TensorSym(x.name, te1.dim)
      val te2 = dimCheck(e2)(ctx + (tx.name -> tx.dim))
      TensorLet(tx, te1, te2)
    case tc @ TensorConst(elems, TensorDim(order, dims)) => 
      if(order != dims.size) 
        raise(s"Order $order doesn't match dims $dims in `$tc`")
      if(elems.size != dims.product) {
        raise(s"Number of elements ${elems.size} doesn't match dims $dims in `$tc`")
      }
      tc
    case TensorSym(name, _) => 
      val dim = ctx.getOrElse(name, raise(s"Symbol `$name` not in the environment: $ctx"))
      TensorSym(name, dim)
    case TensorOnes(name, dim) => 
      TensorOnes(name, dim)
    case TensorLoad(name, args, dim) => 
      TensorLoad(name, args, dim)
    case TensorArithmetic(e1, e2, op) => 
      val e1_ = dimCheck(e1)(ctx).asInstanceOf[TensorSym]
      val e2_ = dimCheck(e2)(ctx).asInstanceOf[TensorSym]
      TensorArithmetic(e1_, e2_, op)  
    case TensorContract(args, einsum, _) =>
      // println(s"args: ${args}")
      val targs = args.map(a => dimCheck(a)(ctx))
      val idims = targs.map(_.dim.dims)
      val temp_str = einsum.split("->")
      var dim = TensorDim.Scalar // Assuming that output tensor is a scalar 
      if (temp_str.length > 1) { // If output tensor is not a scalar, check dims
        val istrs = temp_str(0).split(",") 
        val ostr = temp_str(1)
        val imaps = idims.zip(istrs).map(ds => (ds._2 zip ds._1).toMap) 
        // println(s"imaps: ${imaps}\n")
        // val umaps = imaps.reduce((x, y) => x ++ y) // unify maps
        var umaps = imaps(0)
        if (imaps.length > 1) { // iterate to check dims matching
          for (i <- 1 to imaps.length-1) {
            val compare = imaps(i)
            val share = umaps.keySet.intersect(compare.keySet)
            if (!share.isEmpty) {
              if (!share.map(x => umaps(x) == compare(x)).forall(x => true)) {
                throw new Exception("Dimensions in einsum expression doesn't match.")
              }
            }
            umaps = umaps ++ compare
          }
        }
        dim = TensorDim(ostr.size, ostr.map(c => umaps(c)))
      }
      TensorContract(targs, einsum, dim)
    case TensorSelect(e, filter, filterDim) =>
      TensorSelect(dimCheck(e)(ctx), filter, filterDim)
    case TensorNormalize(e, normDims) =>
      TensorNormalize(dimCheck(e)(ctx), normDims) // FIXME
    case DictionaryLoad(name, dims) =>
      DictionaryLoad(name, dims)
    case TensorReverseMap(e, dicts, dim) =>
      TensorReverseMap(dimCheck(e)(ctx), dicts, dim)
    case TensorLift(e) =>
      TensorLift(dimCheck(e)(ctx))
  }
}