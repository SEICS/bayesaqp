package bayestensor
package ir
package setar

case class SetarProg(rules: Seq[SetarRule], ret: SetarSym)
case class SetarRule(hd: SetarAccess, body: SetarProduct, filter: SetarProduct)

// SetarAccess will be SetarTensor(Seq[SetarIndex])
sealed trait SetarTensor
case class SetarSym(name: String) extends SetarTensor
// case class SetarConstValue(elems: Seq[Double], dim: TensorDim) extends SetarTensor
case class SetarWrapper(texp: TensorExp) extends SetarTensor
case class SetarIndex(name: String)
// Extends Setar syntax for inlining everything

sealed trait SetarExp

// numeric domain
case class SetarAccess(ten: SetarTensor, idcs: Seq[SetarIndex]) extends SetarExp
case class SetarNorm(args: SetarProduct, idcs: Seq[SetarIndex]) extends SetarExp
case class SetarDiv(numerator: SetarProduct, nidcs: Seq[SetarIndex], denominator: SetarProduct, didcs: Seq[SetarIndex]) extends SetarExp

// boolean domain
case class SetarDim(idcs: Seq[SetarIndex], dim: TensorDim) extends SetarExp
case class SetarRange(idx: SetarIndex, rng: Seq[Int]) extends SetarExp
