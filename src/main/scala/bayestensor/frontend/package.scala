package bayestensor
import ir._

package object frontend {
  implicit class Interpolator(val sc: StringContext) {
    def bt(args: Any*): Exp = {
      val strings = sc.parts.iterator
      val expressions = args.iterator
      val buf = new StringBuffer(strings.next)
      while (strings.hasNext) {
        val nextExpression = expressions.next match {
          case b: Boolean => b.toString
          case d: Double => d.toString
          case i: Int => i.toString
          case x =>
            raise(s"doesn't know how to splice `$x`")
        }
        buf.append(nextExpression)
        buf.append(strings.next)
      }
      val prog = BayesTensorParser(buf.toString)
      assert(prog.funs.isEmpty)
      prog.body
    }
    def tensor(args: Any*): TensorExp = {
      val strings = sc.parts.iterator
      val expressions = args.iterator
      val buf = new StringBuffer(strings.next)
      while (strings.hasNext) {
        val nextExpression = expressions.next match {
          case d: Double => d.toString
          case i: Int => i.toString
          case x =>
            raise(s"doesn't know how to splice `$x`")
        }
        buf.append(nextExpression)
        buf.append(strings.next)
      }
      TensorParser(buf.toString)
    }
  }
}