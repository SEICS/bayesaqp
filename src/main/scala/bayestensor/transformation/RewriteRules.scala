package bayestensor
package transformation

import frontend._
import ir._
import setar._
import backend._
import os._
import scala.collection._
import scala.util.control.Breaks._
import java.nio.file.Path
import transformation._
import utils._

object RewriteRules extends (SetarProg => SetarProg) {
  def apply(e: SetarProg): SetarProg = {
    // // // Inlining must be applied after OptimiseNorm, otherwise losing info for OptimiseNorm (i.e., need the rule's head of SetarNorm)
    val newRules = time(optimiseNorm(e.rules), "TransferNormToDiv")
    val optNormProg = time(SetarProg(newRules, e.ret), "formSetarProg")
    val inlinedProg = time(Inline(optNormProg), "Inlining")
    val newRules2 = time(eliminateNorm(inlinedProg.rules), "EliminateNorm")
    val newRules3 = time(newRules2.map(r => mergeDivNumerator(r)), "MergeDivNumerator")
    val newRules4 = time(rewriteOnes(newRules3).filter(r => !r.body.isEmpty), "RewriteOnes")
    val finalRules = time(deadCodeElimination(newRules4), "DeadCodeElimination")
    if (finalRules.isEmpty) {
        e
    } else {
        SetarProg(finalRules, e.ret)
    }
  }
  def collectIdcs(sexp: SetarExp): mutable.Set[SetarIndex] = {
    val uniqueIdcs = mutable.Set[SetarIndex]()
    sexp match {
        case SetarAccess(_, idcs1) => 
            uniqueIdcs ++= idcs1.toSet
        case SetarNorm(args, _) => 
            for (a <- args) {
                uniqueIdcs ++= collectIdcs(a)
            }
        case SetarDiv(numerator, nidcs, denominator, didcs) => 
            for (nd <- numerator ++ denominator) {
                uniqueIdcs ++= collectIdcs(nd)
            }
    }
    uniqueIdcs
  }
  def mergeDivNumerator(rule: SetarRule) = { // // // Merge SetarDiv(C/D)*A*B into SetarDiv(C*A*B/D) to reduce dimensions of intermediate tensors
    val body = rule.body
    val divIdcs = body.zipWithIndex.filter(nb => nb._1 match {
        case s: SetarDiv => true
        case _ => false
    }).map(_._2) // idx in list
    var newBody = mutable.Seq[SetarExp]()
    var newBody2 = mutable.Seq[SetarExp]()
    if (!divIdcs.isEmpty && !(body.length == divIdcs.length)) {
        for (di <- divIdcs) {
            newBody = newBody :+ body(di)
        }
        val nonDivIdcs = body.indices.filterNot(i => divIdcs.contains(i))
        // println(s"nonDivIdcs; ${nonDivIdcs}")
        for (idx <- nonDivIdcs) {
            val nb = body(idx)
            // println(s"nb: ${nb}")
            var divCounter = 0
            var merge = false
            while ((divCounter < newBody.length) && (!merge)) {
                val div = newBody(divCounter).asInstanceOf[SetarDiv]
                // println(s"div: ${div.numerator}")
                if (collectIdcs(nb).&~((div.asInstanceOf[SetarDiv].numerator).map(n => collectIdcs(n)).flatten.toSet).isEmpty) { // able to merge
                    var nidcs2 = div.nidcs
                    merge = true
                    val freeVars = collectIdcs(nb).&~(rule.hd.idcs.toSet)
                    if (!freeVars.isEmpty) { // if free variable
                        val restDiv = newBody.zipWithIndex.filterNot(i => i._2 != divIdcs(divCounter)).map(_._1)
                        val restDivIdcs = restDiv.map(rd => rd.asInstanceOf[SetarDiv].nidcs).flatten
                        if (freeVars.&~(restDivIdcs.toSet).isEmpty) { 
                            nidcs2 = nidcs2.filterNot(ni => freeVars.contains(ni))
                        } 
                    } 
                    newBody(divCounter) = SetarDiv(div.numerator ++ Seq(nb), nidcs2, div.denominator, div.didcs)
                } 
                divCounter += 1
            }
            if (!merge) {
                newBody2 = newBody2 :+ nb
            }
        }
        newBody = newBody ++ newBody2
    } else {
        newBody = newBody ++ body
    }
    SetarRule(rule.hd, newBody, rule.filter)
  }

  def optimiseNorm(rules: Seq[SetarRule]): Seq[SetarRule] = {
    val candidates = mutable.Set[Int]()
    var newRules = Seq[SetarRule]()
    rules.zipWithIndex.map(r => 
        r._1.body.map(b =>
            b match {
                case SetarNorm(_,_) => 
                    candidates += r._2
                    // if (r._1.hd.idcs.size > 2) { // Identify candidates: > 2
                    //     candidates += r._2
                    // }
                    // candidates += r._2
                case SetarAccess(_,_) => "Ignored"
            }
        )
    )
    
    // println(s"candidates:${candidates}")
    for ((r, i) <- rules.zipWithIndex) {
        if (candidates.contains(i)) { // If candidates, rewrite SetarNorm as SetarDiv
            var newBody = Seq[SetarExp]()
            var numerator = Seq[SetarExp]()
            var denominator = Seq[SetarExp]()
            var normDims = Seq[SetarIndex]()
            for ((b,j) <- r.body.zipWithIndex) {
                b match {
                    case SetarNorm(args, idcs) => 
                        numerator ++= args
                        denominator ++= args
                        normDims ++= idcs
                        // val optNorm = SetarDiv(args, args, r.hd.idcs) 
                    case SetarAccess(_,_) => 
                        numerator = numerator :+ b
                        // newBody = newBody :+ b
                }
            }
            val optNorm = SetarDiv(numerator, r.hd.idcs, denominator, normDims) // SetarIndice for head should not be changed
            newBody = newBody :+ optNorm
            val newRule = SetarRule(r.hd, newBody, r.filter)
            newRules = newRules :+ newRule
        } else {
            newRules = newRules :+ r
        }
    }
    newRules
  } 

  def findRemain(sexp: SetarExp, ruleNo: Int, toRemove: mutable.Map[Int, String]): mutable.Set[Int] = {
    val remainLines = mutable.Set[Int]()
    sexp match {
        case SetarAccess(ten,_) =>
            ten match {
                case SetarSym(name) => 
                    var prevLine = ruleNo - 1
                    var found = false
                    while ((prevLine >= 0) && (found == false)) {
                        if (toRemove.contains(prevLine)) {
                            if (toRemove(prevLine) == name) {
                                found = true
                                remainLines += prevLine
                            } 
                        } 
                        prevLine -= 1
                    }
                case SetarWrapper(_) => "Ignored"
            }
        case SetarNorm(args,_) =>
            for (a <- args) {
                val rl = findRemain(a, ruleNo, toRemove)
                remainLines ++= rl
            }
        case SetarDiv(numerator,_,denominator,_) =>
            val vars = numerator ++ denominator
            for (v <- vars) {
                val rl = findRemain(v, ruleNo, toRemove)
                remainLines ++= rl
            }
    }
    remainLines
  }
  def isWrapper(sexp: SetarExp): Boolean = {
    sexp match {
        case SetarAccess(ten, _) =>
            ten match {
                case s: SetarWrapper => true
                case _ => false
            }
        case _ => false
    }
  }

  def getVarsInWrapper(texp: TensorExp, varsInWrapper: mutable.Set[String]): mutable.Set[String] = {
    texp match {
        case TensorSym(name,_) => 
            varsInWrapper += name
        case TensorLift(texp2) =>
            varsInWrapper ++= getVarsInWrapper(texp2, varsInWrapper)
        case TensorReverseMap(texp3,_,_) =>
            varsInWrapper ++= getVarsInWrapper(texp3, varsInWrapper)
        case TensorArithmetic(texp4, texp5, _) =>
            varsInWrapper ++= getVarsInWrapper(texp4, varsInWrapper)
            varsInWrapper ++= getVarsInWrapper(texp5, varsInWrapper)
        case _ => "Ignore"
    }
    varsInWrapper
  }
  def deadCodeElimination(rules: Seq[SetarRule]): Seq[SetarRule] = {
    // var newRules = Seq[SetarRules]()
    val toRemove = mutable.Map[Int, String]()
    val lastNonWrapperRule = rules.zipWithIndex.filter(r => r._1.body.map(b => isWrapper(b)).exists(_ == false)).map(_._2).max
    
    val varsInWrapper = mutable.Set[String]()
    rules.zipWithIndex.dropRight(1).map(r =>
        r._1.body.map(b => 
            b match {
                case SetarAccess(ten, _) =>
                    ten match {
                        case SetarWrapper(texp) => 
                            varsInWrapper ++= getVarsInWrapper(texp, varsInWrapper)
                        case _ => "Ignore"
                    }
                case _ => "Ignore"
            }
        )
    )
    // println(s"varsInWrapper: ${varsInWrapper}")
    rules.zipWithIndex.dropRight(1).map(r => 
        r._1.hd.ten match {
        case SetarSym(name) =>
            // val notLastRule = r._2 < lastNonWrapperRule // not last rule (excluding rules that are wrappers only)
            val notAllWrapper = r._1.body.map(b => isWrapper(b)).exists(_ == false) // rule with only wrappers should not be removed
            if (notAllWrapper && (!(varsInWrapper.contains(name)))) {
                toRemove += (r._2 -> name)
            }
        case x => raise(s"Head of Setar Rule should be SetarSym: `${x}`.")
    }).toSet
    val remainLines = mutable.Set[Int]()
    for ((rule, ruleNo) <- rules.zipWithIndex) {
        for (b <- rule.body) {
            val rl = findRemain(b, ruleNo, toRemove)
            remainLines ++= rl
        }
    }
    for (rl <- remainLines) {
        toRemove -= rl
    }
    rules.zipWithIndex.filterNot(r => toRemove.contains(r._2)).map(_._1)
  }
  def getSetarTensor(sexp: SetarExp): mutable.Set[SetarTensor] = {
    val unique = mutable.Set[SetarTensor]()
    sexp match {
        case SetarAccess(ten, _) => unique += ten
        case SetarNorm(args, _) => 
            args.map(a =>
                unique ++= getSetarTensor(a)     
            )
        case SetarDiv(_, _, denominator, didcs) =>
            denominator.map(d =>
                unique ++= getSetarTensor(d)     
            )
    }
    unique
  }
  def getBodyST(sp: SetarProduct): mutable.Set[SetarTensor] = {
    val unique = mutable.Set[SetarTensor]()
    sp.map(spp => 
        unique ++= getSetarTensor(spp)
    )
    unique
  }
  def genEinsumSymbol(counter: Int) = SetarIndex((97+counter).toChar.toString)
  def renameSexpIdcs(sexp: SetarExp, renameMap: Map[SetarIndex, SetarIndex]): SetarExp = {
    sexp match {
        case SetarAccess(ten, idcs) => 
            SetarAccess(ten, idcs.map(i =>
                if (renameMap.contains(i)) {
                    renameMap(i)
                } else {
                    i
                }
            ))
        case SetarNorm(args, idcs) =>
            SetarNorm(args.map(a => renameSexpIdcs(a, renameMap)), idcs.map(i =>
                if (renameMap.contains(i)) {
                    renameMap(i)
                } else {
                    i
                }
            ))
        case b3 =>
            raise(s"This type `${b3}` is not supported for renaming SetarIndex in inlining.")
    }
  }
  def isNorm(sexp: SetarExp): Boolean = {
    sexp match {
        case s: SetarNorm => true
        case s2: SetarDiv => true
        case _ => false
    }
  }
  def eliminateNorm(rules: Seq[SetarRule]): Seq[SetarRule] = {
    val norm = mutable.Map[SetarTensor, Seq[SetarProduct]]() // SeterAccess: (Norm, unNorm)
    val unnorm = mutable.Map[SetarTensor, SetarProduct]() // SeterAccess: unNorm'
    val normMap = mutable.Map[SetarTensor, mutable.Map[Int, SetarProduct]]()
    var counter = 0 // For generating new einsum symbol
    for ((rule, i) <- rules.zipWithIndex) {
        val ht = getSetarTensor(rule.hd).toSeq(0)
        // val hdSym = rule.hd.ten.asInstanceOf[SetarSym] 
        if (normMap.contains(ht)) {
            normMap(ht) += (i -> rule.body)
            // inlineMap(hdSym) += (i -> rule)
        } else {
            normMap += (ht -> mutable.Map[Int, SetarProduct](i -> rule.body))
            // inlineMap += (hdSym -> mutable.Map[Int, SetarRule](i -> rule))
        }
        val existsNorm = rule.body.map(b => 
            b match {
                case b1: SetarNorm => true
                case _ => false
            }    
        )
        if (existsNorm.exists(n => n == true)) { // Collect all rules' bodiess that cohtains Norm(X)s
            rule.body.map(b => 
                b match {
                    case SetarNorm(ten, _) =>
                        if (norm.contains(ht)) {
                            norm(ht) = Seq(norm(ht)(0) ++ ten, norm(ht)(1))
                        } else {
                            norm += (ht -> Seq(ten, Seq()))
                        }
                    case SetarDiv(numerator, nidcs, denominator, didcs) => 
                        if (norm.contains(ht)) {
                            norm(ht) = Seq(norm(ht)(0) ++ denominator, norm(ht)(1))
                        } else {
                            norm += (ht -> Seq(denominator, Seq()))
                        }
                    case _ => 
                        if (norm.contains(ht)) {
                            norm(ht) = Seq(norm(ht)(0), norm(ht)(1) :+ b)
                        } else {
                            norm += (ht -> Seq(Seq(), Seq(b)))
                        }
                }
            )
        } else { // Collect all rules' bodies that don't cohtain Norm(X)
            if (unnorm.contains(ht)) {
                unnorm(ht) = unnorm(ht) ++ rule.body
            } else {
                unnorm += (ht -> rule.body)
            }
        }
    }
    // println(s"normMap: ${normMap}")
    // println(s"norm: ${norm}")
    // println(s"unnorm: ${unnorm}")
    val candidates = mutable.Map[Int, SetarProduct]() // Mark candidate for removing Norm
    val inlineMap = mutable.Map[SetarSym, mutable.Map[Int, SetarRule]]() // Int referring to nth line of code
    for ((rule, i) <- rules.zipWithIndex) {
        val bodyWithIdx = rule.body.zipWithIndex
        for ((b, j) <- bodyWithIdx) {
            val rest = bodyWithIdx.filter(_._2 != j).map(_._1)
            val restNames = rest.map(x =>
                x match {
                    case SetarAccess(ten, _) =>
                        ten match {
                            case SetarSym(name) => name
                            case _ => ""
                        }
                    case normOrDiv @ (SetarNorm(_,_) | SetarDiv(_,_,_,_)) => ""
                }
            ).filter(y => y != "")
            b match {
                case SetarAccess(ten, _) =>
                    ten match {
                        case s1: SetarSym =>
                            val isCand1 = norm.contains(s1) 
                            val isCand2 = restNames.exists(x => norm.contains(s1))
                            if (isCand1 || isCand2) {
                                if (candidates.contains(i)) {
                                    candidates(i) = candidates(i) :+ b
                                } else {
                                    candidates += (i -> Seq(b))
                                }
                                
                                if (inlineMap.contains(s1)) {
                                    inlineMap(s1) += (i -> rule)
                                } else {
                                    inlineMap += (s1 -> mutable.Map[Int, SetarRule](i -> rule))
                                }
                            } 
                        case SetarWrapper(_) => "Ignored"
                    }   
                case normOrDiv @ (SetarNorm(_,_) | SetarDiv(_,_,_,_))  => 
                    if (candidates.contains(i)) {
                        candidates(i) = candidates(i) :+ b
                    } else {
                        candidates += (i -> Seq(b))
                    }
            }    
        }
    }
    // println(s"candidates: ${candidates}\n")
    // println(s"SetarCodeGenerator.codegen():\n${SetarCodeGenerator.codegen(SetarProg(rules, SetarSym("res")))}\n")
    // println(s"rules: ${rules}")
    var newRules = Seq[SetarRule]()
    for ((rule, k) <- rules.zipWithIndex) {
        if (candidates.contains(k)) { // Needs to rewrite
            // println(s"rule: ${rule}")
            val candBody = candidates(k) // rule's body of this candidate
            var normContent = Seq[SetarProduct]()
            // println(s"candBody: ${candBody}")
            candBody.map(c => c match {
                case SetarAccess(c1, _) =>
                    if (norm.contains(c1)) {
                        normContent = normContent :+ norm(c1)(0)
                    }
                case SetarNorm(args, _) => 
                    normContent = normContent :+ args
                case SetarDiv(numerator, nidcs, denominator, didcs) =>
                    normContent = normContent :+ denominator
            })
            // println(s"normContent: ${normContent}")
            var newBody = Seq[SetarExp]()
            for (nc <- normContent) {
                // println(s"nc: ${nc}")
                val candST = getBodyST(nc)
                // var candSet = Set[SetarIndex]()
                val cond1 = rule.body.map(b => isNorm(b)).exists(_ == true) // any norm exists?
                // val st = getBodyST(nc)
                val b = getBodyST(rule.body)
                // val cond2 = st.&~(b).isEmpty // Is the case: r0 = Norm(A*B*C) && r1 = r0*A*B*C*D... ?
                // r0 = Norm(A*B*C)*E && r1 = A*B*C*D... && r2 = r0*r1 => lead to wrong result if r2 is rewritten as r2 = A*B*C*D*...*E
                if (cond1) { // || (!cond1 && !cond2) // Rule = Norm(A*B*C) * A*B*C*D => Rule = A*B*C*D
                    // println(s"case 1")
                    // println(s"thisRule: ${thisRule}")
                    rule.body.zipWithIndex.map(b =>
                        b._1 match {
                            case SetarNorm(args, _) => 
                                val st = getBodyST(args) 
                                val rest = rule.body.zipWithIndex.filterNot(bb => bb._2 == b._2).map(_._1)
                                // println(s"rest: ${rest}")
                                val cancelNorm = (candST.&~(st).isEmpty) && (candST.&~(getBodyST(rest)).isEmpty)
                                if (!cancelNorm) {
                                    newBody = newBody :+ b._1
                                } 
                            case SetarDiv(numerator, nidcs, denominator, didcs) => 
                                val st = getBodyST(denominator) 
                                val rest = rule.body.zipWithIndex.filterNot(bb => bb._2 == b._2).map(_._1)
                                // println(s"rest: ${rest}\n")
                                // println(s"candST.&~(st): ${candST.&~(st)}")
                                // println(s"candST.&~(getBodyST(rest)): ${candST.&~(getBodyST(rest))}")
                                // println(s"st: ${st}\n")
                                // println(s"candST: ${candST}\n")
                                val cancelNorm = (candST.&~(st).isEmpty) && (candST.&~(getBodyST(rest)).isEmpty)
                                // println(s"cancelNorm: ${cancelNorm}")
                                if (!cancelNorm) {
                                    newBody = newBody :+ b._1
                                } else {
                                    newBody = newBody ++ numerator.filterNot(bb => getSetarTensor(bb).&~(candST).isEmpty)
                                }
                                // else {
                                //     newBody = newBody ++ rest.filterNot(bb => getSetarTensor(bb).&~(candST).isEmpty)
                                // }
                            case s: SetarAccess => newBody = newBody :+ s
                        }
                    )
                } else { // In this case, the body of this rule should only contains SetarAccess
                    val st = getBodyST(nc)
                    val b = getBodyST(rule.body)
                    // println(s"st.&~(b).isEmpty: ${st.&~(b).isEmpty}")
                    if (st.&~(b).isEmpty) { // possibility: r0 = Norm(A*B*C) && r1 = r0*A*B*C*D...
                        // println(s"case 2")
                        val potentialNorm = rule.body.filter(pn => getSetarTensor(pn).&~(b.&~(st)).isEmpty) // will be r0*D*... in r1
                        newBody = newBody ++ rule.body.filter(pn => !getSetarTensor(pn).&~(b.&~(st)).isEmpty) // A*B*C should be remained
                        // println(s"potentialNorm: ${potentialNorm}")
                        for (pn <- potentialNorm) {
                            pn match {
                                case SetarAccess(s1, _) => 
                                    if (normMap.contains(s1)) {
                                        val prevLine = normMap(s1).keys.filter(x => x < k).max
                                        val pnb = normMap(s1)(prevLine)
                                        // println(s"pnb: ${pnb}\n")
                                        pnb.map(pnbb => 
                                            if (isNorm(pnbb) && st.&~(getSetarTensor(pnbb)).isEmpty) { // Yes, verify that r0 is the norm candidate that needs to be removed. Then remove r0
                                                newBody ++= Seq()
                                            } else {
                                                newBody = newBody :+ pnbb
                                            }
                                        )
                                    }
                                case s2: SetarNorm => raise(s"`$s2` should be captured before this step and eliminated.")
                                case s3: SetarDiv => raise(s"`$s3` should be captured before this step and eliminated.")
                            }
                        }
                    } else { // r0 = Norm(A*B*C)*E && r1 = A*B*C*D... && r2 = r0*r1 => lead to wrong result if r2 is rewritten as r2 = A*B*C*D*...*E
                        // newBody = newBody :+ s4
                        // println(s"case 3")
                        if (newBody.isEmpty) {
                            newBody = newBody ++ rule.body
                        }
                        // var toRemain = Seq[SetarExp]()
                        // println(s"rule.body: ${rule.body}")
                        // // val inlinedRule = Inline.inlineRule(rule, k, inlineMap) // In this case, inlining r0 and r1 into r2
                        // for ((v, vi) <- rule.body.zipWithIndex) { // make one pass to see if r1 can cancel the norm r0
                        //     val rest = rule.body.zipWithIndex.filterNot(rb => rb._2 == vi).map(_._1)
                        //     // println(s"v: ${v}")
                        //     v match {
                        //         case SetarAccess(s3, idcs) => 
                        //             if (normMap.contains(s3)) {
                        //                 val prevLine = normMap(s3).keys.filter(x => x < k).max
                        //                 val pnb = normMap(s3)(prevLine)
                        //                 val prevIdcs = rules(prevLine).hd.idcs.toSet
                        //                 val (pnb2, count) = inlineSymbolMatch(pnb, rest, prevIdcs, idcs.toSet, counter)
                        //                 counter += count
                        //                 println(s"pnb2: ${pnb2}")
                        //                 pnb2.map(pnbb => pnbb match {
                        //                         // case SetarNorm(args, _) => 
                        //                         //     val cond3 = st.&~(getBodyST(args)).isEmpty
                        //                         //     // println(s"cond3: ${cond3}")
                        //                         //     if (!cond3) {
                        //                         //         newBody = newBody :+ v
                        //                         //     }
                        //                         // case s4: SetarAccess => 
                        //                         //     val inlinedRule = Inline.inlineRule(rule, k, inlineMap)
                        //                         //     // println(s"inlinedRule: ${inlinedRule}\n")
                        //                         //     val cond4 = st.&~(getSetarTensor(s4)).isEmpty
                        //                         //     if (!cond4) { // yes, found r1s
                        //                         //         newBody = newBody :+ s4
                        //                         //     } 
                        //                         case SetarNorm(args, _) => 
                        //                             val cond3 = st.&~(getBodyST(args)).isEmpty
                        //                             // println(s"cond3: ${cond3}")
                        //                             if (!cond3) {
                        //                                 newBody = newBody :+ v
                        //                             } 
                        //                         case s4: SetarAccess => 
                        //                             val cond4 = st.&~(getSetarTensor(s4)).isEmpty
                        //                             if (!cond4) { // yes, found r1s
                        //                                 newBody = newBody :+ s4
                        //                             }
                        //                     }
                        //                 ) 
                        //             }
                        //         case s4: SetarNorm => raise(s"SetarNorm `$s4` should be handled by other cases.")
                        //     }
                            
                        // }
                    }
                }
            }
            // println(s"newRule:\n${SetarRule(rule.hd, newBody, rule.filter)}\n")
            val newRule = SetarRule(rule.hd, newBody, rule.filter)
            normMap(getSetarTensor(rule.hd).toSeq(0))(k) = newBody
            newRules = newRules :+ newRule
        } else { // Keep the origin
            newRules = newRules :+ rule
        }
    }
    newRules
  }
  def rewriteOnes(rules: Seq[SetarRule]): Seq[SetarRule] = {
    // Remove TensorOnes
    // val hdToBody = mutable.Map[SetarSym, SetarProduct]()
    // rules.map(r => 
    //     hdToBody += (r.hd.ten.asInstanceOf[SetarSym] -> r.body)
    // )
    // println(s"hdToBody: ${hdToBody}")
    var newRules = Seq[SetarRule]()
    val toRemove = mutable.Set[SetarTensor]()
    for (rule <- rules) { 
        var newBody = Seq[SetarExp]()
        for (b <- rule.body) {
            // println(s"toRemove: ${toRemove}")
            val (nb, toRemove2) = removeOnes(rule, b, toRemove)
            toRemove ++= toRemove2
            newBody ++= nb
        }
        newRules = newRules :+ SetarRule(rule.hd, newBody, rule.filter)
    }
    newRules
  }
  def removeOnes(rule: SetarRule, sexp: SetarExp, toRemove: mutable.Set[SetarTensor]): (SetarProduct, mutable.Set[SetarTensor]) = {
    sexp match {
        case SetarAccess(ten, _) => 
            if (toRemove.contains(ten)) {
                (Seq(), toRemove)
            } else {
                ten match {
                    case SetarSym(_) => 
                        (Seq(sexp),toRemove)
                    case SetarWrapper(texp) =>
                        texp match {
                            case t1: TensorOnes => // Remove ones for computing probability of COUNT
                                if (rule.body.length == 1) {
                                    toRemove += rule.hd.asInstanceOf[SetarAccess].ten.asInstanceOf[SetarSym]
                                }
                                (Seq(), toRemove)
                            case _ =>
                                (Seq(sexp), toRemove)
                        }
                }
            } 
        case SetarNorm(args, idcs) =>
            var newArgs = Seq[SetarExp]()
            for (a <- args) {
                val (a2, toRemove2) = removeOnes(rule, a, toRemove)
                toRemove ++= toRemove2
                newArgs = newArgs ++ a2
            }
            (Seq(SetarNorm(newArgs, idcs)), toRemove)
        case SetarDiv(numerator, nidcs, denominator, didcs) =>
            var newArgs1 = Seq[SetarExp]()
            for (n <- numerator) {
                val (a3, toRemove3) = removeOnes(rule, n, toRemove)
                toRemove ++= toRemove3
                newArgs1 = newArgs1 ++ a3
            }
            var newArgs2 = Seq[SetarExp]()
            for (n <- denominator) {
                val (a4, toRemove4) = removeOnes(rule, n, toRemove)
                toRemove ++= toRemove4
                newArgs2 = newArgs2 ++ a4
            }
            (Seq(SetarDiv(newArgs1, nidcs, newArgs2, didcs)), toRemove)
    }
  }
  def inlineSymbolMatch(pnb: SetarProduct, rest: SetarProduct, prevIdcs: Set[SetarIndex], idcs: Set[SetarIndex], counter: Int): (SetarProduct, Int) = {
    var updatedCount = counter
    var freeVars = Set[SetarIndex]()
    pnb.map(sp => 
        sp match {
            case SetarAccess(_, idcs1) => 
                freeVars ++= idcs1.toSet
            case SetarNorm(_, idcs2) => 
                freeVars ++= idcs2.toSet
        }   
    )
    
    freeVars = freeVars.diff(prevIdcs).intersect(idcs.toSet)
    var renameMap = Map[SetarIndex, SetarIndex]()
    var candSet = Set[SetarIndex]() // currHead.idcs.toSet
    for (cb <- rest) {
        cb match {
            case SetarAccess(_, idcs1) => 
                candSet ++= idcs1.toSet
            case SetarNorm(_, idcs2) => 
                candSet ++= idcs2.toSet
        }
    }
    if (!freeVars.isEmpty) {
        for (tr <- freeVars) {
            var newSymbol = genEinsumSymbol(updatedCount)
            val nonUnique = freeVars.diff(prevIdcs) ++ idcs.toSet ++ candSet ++ renameMap.values.toSet 
            while (nonUnique.contains(newSymbol)) { // Generate symbol until it is unique
                updatedCount += 1
                newSymbol = genEinsumSymbol(updatedCount)
            }
            renameMap += (tr -> newSymbol)
        }
    }
    val pnb1 = pnb.map(v => renameSexpIdcs(v, renameMap))
    val symbolMatch = (prevIdcs zip idcs).toMap
    val pnb2 = pnb1.map(v => renameSexpIdcs(v, symbolMatch))
    (pnb2, updatedCount)
  }
}
