package bayestensor
package transformation

import frontend._
import ir._
import setar._
import backend._
import os._
import scala.collection._
import java.nio.file.Path


object Inline extends (SetarProg => SetarProg) {
  def apply(e: SetarProg): SetarProg = { // Top-down Inlining
    val inlineCheckSet = mutable.Set[Int]()
    val inlineMap = mutable.Map[SetarSym, mutable.Map[Int, SetarRule]]() // Int referring to nth line of code
    val normTensor = mutable.Set[SetarTensor]()
    val inlineNonNorm = mutable.Set[SetarTensor]()
    val lastNonWrapperRule = e.rules.zipWithIndex.filter(r => r._1.body.map(b => isWrapper(b)).exists(_ == false)).map(_._2).max
    // println(s"lastNonWrapperRule: ${lastNonWrapperRule}")
    // val frequency = count_freq(e.rules)
    // println(s"frequency: ${frequency}")
    val varsInWrapperSet = mutable.Set[String]()
    val varsBodyInWrapperSet = mutable.Set[SetarSym]()
    for (r <- e.rules) {
        for (b <- r.body) {
            val (vb, vi) = getVarsInWrapper(b, r.hd, varsBodyInWrapperSet, varsInWrapperSet)
            varsBodyInWrapperSet ++= vb
            varsInWrapperSet ++= vi
        }
    }
    // println(s"varsBodyInWrapperSet: ${varsBodyInWrapperSet}")
    // println(s"varsInWrapperSet: ${varsInWrapperSet}")
    e.rules.zipWithIndex.map(r => 
        if (r._1.hd.idcs.size > 2) { // Candidate for inlining
            r._1.hd.ten match {
                case t: SetarSym => 
                    // println(s"r._1.body.map(b => isNorm(b)).forall(x => x == false): ${r._1.body.map(b => isNorm(b)).forall(x => x == false)}")
                    // println(s"codeNum: ${r._2}\n")
                    // println(s"r._1.body(0) match {case x: SetarWrapper => true}: ${r._1.body(0) match {case x: SetarWrapper => true}}")
                    // if (r._2 < e.rules.size - 1) { // last rule should not be removed
                    //     if (inlineMap.contains(t)) {
                    //         inlineMap(t) += (r._2 -> r._1)
                    //     } else {
                    //         inlineMap += (t -> mutable.Map[Int, SetarRule](r._2 -> r._1))
                    //     }
                    //     inlineCheckSet += r._2 // Mark to be inlined
                    // }
                    val notLastRule = r._2 < lastNonWrapperRule // not last rule (excluding rules that are wrappers only)
                    val notAllWrapper = r._1.body.map(b => isWrapper(b)).exists(_ == false) // rule with only wrappers cannot be inlined
                    // println(s"${r._1.hd.ten}: ${notAllWrapper}")
                    
                    // println(s"${r._1.hd.ten} codeNum ${r._2}: ${ifBodySymInWrapper}")
                    // if (bodySymInWrapper) {
                    //     println(s"r._1.body: ${r._1.body}")
                    // }
                    
                    // println(s"${r._2} ${r._1}: ${r._1.body.map(b => isWrapper(b))}")
                    if (notLastRule && notAllWrapper) { // last rule should not be removed
                        if (!checkIfAnySymBodyInWrapper(r._1, t, varsBodyInWrapperSet)) { // !varsBodyInWrapperSet.contains(t)
                            // println(s"${r._1.hd.ten} codeNum ${r._2}")
                            if (inlineMap.contains(t)) {
                                inlineMap(t) += (r._2 -> r._1)
                            } else {
                                inlineMap += (t -> mutable.Map[Int, SetarRule](r._2 -> r._1))
                            }
                            // inlineCheckSet += r._2 
                            if (!varsInWrapperSet.contains(t.name)){
                                inlineCheckSet += r._2 
                            }
                        } 
                        // if (!varsInWrapperSet.contains(t.name) && !varsBodyInWrapperSet.contains(t)) { // if vars is also no longer needed in SetarWrapper
                        //     inlineCheckSet += r._2 // Mark to be inlined
                        // }
                    } 
                case _ =>
                    raise(s"`${r._1.hd.ten}`: Head of a SetarRule should be a SetarSym.")
            }
        } 
    )
    // println(s"inlineNonNorm: ${inlineNonNorm}")
    // println(s"inlineMap: ${inlineMap}")
    var newRules = Seq[SetarRule]()
    for ((r, idx) <- e.rules.zipWithIndex) {
        val (newRule, inlineMap2) = inlineRule(r, idx, inlineMap, varsBodyInWrapperSet)
        newRules :+= newRule
        inlineMap ++= inlineMap2
    }
    // println(s"inlineCheckSet: $inlineCheckSet")
    val finalRules = newRules.zipWithIndex.filter(ri => !inlineCheckSet.contains(ri._2)).map(ri => ri._1)
    // println(s"inlinedProg:\n${SetarCodeGenerator.codegen(SetarProg(newRules, e.ret))}\n")
    // val finalRules = newRules2.map(r => mergeDivNumerator(r))
    SetarProg(finalRules, e.ret)
  }
  def checkIfAnySymBodyInWrapper(rule: SetarRule, hd: SetarSym, varsBodyInWrapperSet: mutable.Set[SetarSym]): Boolean = {
    val ifInWrapper = mutable.Set[Boolean]()
    for (b <- rule.body) {
        val (iw, vb) = checkIfEachBodyInWrapper(b,hd,ifInWrapper,varsBodyInWrapperSet)
        ifInWrapper ++= iw
        varsBodyInWrapperSet ++= vb
    }
    ifInWrapper.exists(_ == true)
  }
  def checkIfEachBodyInWrapper(sexp: SetarExp, hd: SetarSym, ifInWrapper: mutable.Set[Boolean], varsBodyInWrapperSet: mutable.Set[SetarSym]): (mutable.Set[Boolean], mutable.Set[SetarSym]) = {
    sexp match {
        case SetarAccess(ten, _) => 
            ten match {
                case s: SetarSym => 
                    if (varsBodyInWrapperSet.contains(s)) {
                        varsBodyInWrapperSet += hd
                        ifInWrapper += true
                        (ifInWrapper, varsBodyInWrapperSet)
                    } else {
                        ifInWrapper += false
                    }
                    (ifInWrapper, varsBodyInWrapperSet)
                case s: SetarWrapper => (ifInWrapper ++ Seq(true), varsBodyInWrapperSet)
            }
        case SetarDiv(numerator, _, denominator, _) =>
            for (nd <- (numerator.toSet ++ denominator.toSet)) {
                val (res, wrapperSet) = checkIfEachBodyInWrapper(nd, hd, ifInWrapper, varsBodyInWrapperSet)
                varsBodyInWrapperSet ++= wrapperSet
                ifInWrapper ++= res
            }
            (ifInWrapper, varsBodyInWrapperSet)
        case SetarNorm(args, _) => 
            for (a <- args) {
                val (res, wrapperSet) = checkIfEachBodyInWrapper(a, hd, ifInWrapper, varsBodyInWrapperSet)
                varsBodyInWrapperSet ++= wrapperSet
                ifInWrapper ++= res
            }
            (ifInWrapper, varsBodyInWrapperSet)
    }
  }
  def getVarsInWrapper(sexp: SetarExp, hd: SetarAccess, varsBodyInWrapperSet: mutable.Set[SetarSym], varsInWrapperSet: mutable.Set[String]): (mutable.Set[SetarSym], mutable.Set[String]) = {
    sexp match {
        case SetarAccess(ten, _) =>
            ten match {
                case SetarWrapper(texp) => 
                    texp match {
                        // case TensorLoad(name, _, _) =>
                        //     varsInWrapperSet += name
                        // case TensorOnes(name, _) =>
                        //     varsInWrapperSet += name
                        // case DictionaryLoad(name, _) =>
                        //     varsInWrapperSet += name
                        case TensorLift(e) =>
                            e match {
                                case TensorSym(name, _) => 
                                    varsInWrapperSet += name
                                    varsBodyInWrapperSet += hd.ten.asInstanceOf[SetarSym]
                                case _ => raise(s"Doesn't support check for `$e`.")
                            }
                        case TensorReverseMap(e, _, _) =>
                            e match {
                                case TensorSym(name, _) => 
                                    varsInWrapperSet += name
                                    varsBodyInWrapperSet += hd.ten.asInstanceOf[SetarSym]
                                case _ => raise(s"Doesn't support check for `$e`.")
                            }
                        case TensorArithmetic(e1, e2, _) =>
                            varsInWrapperSet += e1.name
                            varsInWrapperSet += e2.name
                            varsBodyInWrapperSet += hd.ten.asInstanceOf[SetarSym]
                        case _ => "Ignore"
                    }
                case _ => "Ignore"
            }
        case SetarNorm(args, _) => 
            for (a <- args) {
                val (vb, vi) = getVarsInWrapper(a, hd, varsBodyInWrapperSet, varsInWrapperSet)
                varsBodyInWrapperSet ++= vb
                varsInWrapperSet ++= vi
            }
        case SetarDiv(numerator, _, denominator, _) =>
            for (nd <- (numerator.toSet ++ denominator.toSet)) {
                val (vb, vi) = getVarsInWrapper(nd, hd, varsBodyInWrapperSet, varsInWrapperSet)
                varsBodyInWrapperSet ++= vb
                varsInWrapperSet ++= vi
            }
    }
    (varsBodyInWrapperSet, varsInWrapperSet)  
  }

  def count_freq(rules: Seq[SetarRule]): mutable.Map[SetarSym, Int] = {
    val frequency = mutable.Map[SetarSym, Int]()
    rules.map(r =>
        r.body.map(b =>
            frequency ++= update_freq(b, frequency)
        )
    )
    frequency
  }
  def update_freq(sexp: SetarExp, frequency: mutable.Map[SetarSym, Int]): mutable.Map[SetarSym, Int] = {
    sexp match {
        case SetarAccess(ten, _) =>
            ten match {
                case s: SetarSym =>
                    if (frequency.contains(s)) {
                        frequency(s) += 1
                    } else {
                        frequency += (s -> 1)
                    }
                case _ => "Ignore for now"
            }
        case SetarDiv(numerator, _, denominator, _) => 
            for (nd <- (numerator.toSet ++ denominator.toSet)) {
                frequency ++= update_freq(nd, frequency)
            }
        case SetarNorm(args, _) =>
            for (a <- args) {
                frequency ++= update_freq(a, frequency)
            }
    }
    frequency
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
//   def mergeDivNumerator(rule: SetarRule) = { // // // Merge SetarDiv(C/D)*A*B into SetarDiv(C*A*B/D) to reduce dimensions of intermediate tensors
//     val body = rule.body
//     val divIdcs = body.zipWithIndex.filter(nb => nb._1 match {
//         case s: SetarDiv => true
//         case _ => false
//     }).map(_._2) // idx in list
//     var newBody = mutable.Seq[SetarExp]()
//     var newBody2 = mutable.Seq[SetarExp]()
//     if (!divIdcs.isEmpty && !(body.length == divIdcs.length)) {
//         for (di <- divIdcs) {
//             newBody = newBody :+ body(di)
//         }
//         val nonDivIdcs = body.indices.filterNot(i => divIdcs.contains(i))
//         // println(s"nonDivIdcs; ${nonDivIdcs}")
//         for (idx <- nonDivIdcs) {
//             val nb = body(idx)
//             println(s"nb: ${nb}")
//             var divCounter = 0
//             var merge = false
//             while ((divCounter < newBody.length) && (!merge)) {
//                 val div = newBody(divCounter).asInstanceOf[SetarDiv]
//                 println(s"div: ${div.numerator}")
//                 if (collectIdcs(nb).&~((div.asInstanceOf[SetarDiv].numerator).map(n => collectIdcs(n)).flatten.toSet).isEmpty) { // able to merge
//                     var nidcs2 = div.nidcs
//                     merge = true
//                     val freeVars = collectIdcs(nb).&~(rule.hd.idcs.toSet)
//                     if (!freeVars.isEmpty) { // if free variable
//                         val restDiv = newBody.zipWithIndex.filterNot(i => i._2 != divIdcs(divCounter)).map(_._1)
//                         val restDivIdcs = restDiv.map(rd => rd.asInstanceOf[SetarDiv].nidcs).flatten
//                         if (freeVars.&~(restDivIdcs.toSet).isEmpty) { 
//                             nidcs2 = nidcs2.filterNot(ni => freeVars.contains(ni))
//                         } 
//                     } 
//                     newBody(divCounter) = SetarDiv(div.numerator ++ Seq(nb), nidcs2, div.denominator, div.didcs)
//                 } 
//                 divCounter += 1
//             }
//             if (!merge) {
//                 newBody2 = newBody2 :+ nb
//             }
//         }
//         newBody = newBody ++ newBody2
//     } else {
//         newBody = newBody ++ body
//     }
//     SetarRule(rule.hd, newBody, rule.filter)
//   }
  def getSetarTensor(sexp: SetarExp): mutable.Set[SetarTensor] = {
    val unique = mutable.Set[SetarTensor]()
    sexp match {
        case SetarAccess(ten, _) => unique += ten
        case SetarNorm(args, _) => 
            args.map(a =>
                unique ++= getSetarTensor(a)     
            )
        case SetarDiv(numerator, _, _, _) =>
            (numerator).map(n =>
                unique ++= getSetarTensor(n)     
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
        case SetarDiv(numerator, nidcs, denominator, didcs) => 
            val renamedNumerator = numerator.map(n => renameSexpIdcs(n, renameMap))
            val renamedDenominator = denominator.map(d => renameSexpIdcs(d, renameMap))
            SetarDiv(renamedNumerator, nidcs.map(i =>
                if (renameMap.contains(i)) {
                    renameMap(i)
                } else {
                    i
                }
            ), renamedDenominator, didcs.map(i =>
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
  def genInlinedSexp(sexp: SetarExp, idx: Int, currRule: SetarRule, codeNum: Int, counter: Int, inlineMap: mutable.Map[SetarSym, mutable.Map[Int, SetarRule]]): (SetarProduct, Int, mutable.Map[SetarSym, mutable.Map[Int, SetarRule]]) = {
    var updatedCount = 0
    sexp match {
        case SetarNorm(args, idcs) => 
            var finalSexp = Seq[SetarExp]()
            for ((arg, i) <- args.zipWithIndex) {
                val (sprod, counter2, inlineMap2) = genInlinedSexp(arg, i, currRule, codeNum, updatedCount, inlineMap)
                finalSexp ++= sprod
                updatedCount += counter2
                inlineMap ++= inlineMap2 
            }
            (Seq(SetarNorm(finalSexp, idcs)), updatedCount, inlineMap)
        case SetarDiv(numerator, nidcs, denominator, didcs) =>
            var finalNumerator = Seq[SetarExp]()
            var finalDenominator = Seq[SetarExp]()
            for ((n, i) <- numerator.zipWithIndex) {
                val (sprod1, counter2, inlineMap2) = genInlinedSexp(n, i, currRule, codeNum, updatedCount, inlineMap)
                finalNumerator ++= sprod1
                finalDenominator ++= sprod1
                updatedCount += counter2
                inlineMap ++= inlineMap2 
            }
            // Wrong
            (Seq(SetarDiv(finalNumerator, nidcs, finalDenominator, didcs)), updatedCount, inlineMap)
        case SetarAccess(ten, idcs) =>  
            if (idcs.size > 2) { // Inline tensors that have order > 2
                ten match {
                    case ten1: SetarSym => 
                        if (inlineMap.contains(ten1)) { // Appear in previous code line, do inlining
                            val prevLine = inlineMap(ten1).keys.filter(k => k < codeNum).max // last recent occurence for the inlined tensor variable
                            val prevRule = inlineMap(ten1)(prevLine)

                            // println(s"prevRule: ${prevRule}")
                            // println(s"norm?: ${prevRule.body.map(b => isNorm(b)).forall(x => x == false)}\n")
                            // if (prevRule.body.map(b => isNorm(b)).forall(x => x == false)) {
                            // println(s"ten1: ${ten1} idcs: ${idcs}\nCodeNum: ${codeNum}\nprevLine: ${prevLine}")
                            // println(s"prevRule: ${prevRule}")
                            // println(s"currRule: ${currRule}")
                            
                            // Now, match the symbols between the previous head and the new body
                            // Check if we need to rename a SetarIndex first: Below code also rename free variables that share the same SetarIndex in currHead
                            val prevSet = mutable.Set[SetarIndex]()
                            for (pb <- prevRule.body) {
                                prevSet ++= collectIdcs(pb)
                            }
                            val currSet = mutable.Set[SetarIndex]() // currHead.idcs.toSet
                            for (cb <- currRule.body) {
                                currSet ++= collectIdcs(cb)
                            }
                            val freeVars = prevSet.&(currSet) // Detect free variables 
                            // println(s"freeVars: ${freeVars}")
                            var renameMap = Map[SetarIndex, SetarIndex]()
                            if (!freeVars.isEmpty) {
                                for (tr <- freeVars) {
                                    var newSymbol = genEinsumSymbol(updatedCount)
                                    val nonUnique = prevSet ++ currSet ++ renameMap.values.toSet ++ freeVars
                                    // println(s"nonUnique: ${nonUnique}")
                                    while (nonUnique.contains(newSymbol)) { // Generate symbol until it is unique
                                        updatedCount += 1
                                        newSymbol = genEinsumSymbol(updatedCount)
                                    }
                                    renameMap += (tr -> newSymbol)
                                }
                            }
                            // println(s"renameMap: ${renameMap}")
                            val prevRuleBody = prevRule.body.map(b => renameSexpIdcs(b, renameMap)) // Rename free variables for rule's body
                            val prevHead = renameSexpIdcs(prevRule.hd, renameMap).asInstanceOf[SetarAccess] // Rename free variables for rule's head
                            val prevFilter = prevRule.filter.map(f => renameSexpIdcs(f, renameMap)) // Rename free variables for rule's filters
                            val newPrevRule = SetarRule(prevHead, prevRuleBody, prevFilter)
                            val symbolMatch = (prevHead.idcs zip idcs).toMap // Backtrack and replace the SetarIndex
                            // if (codeNum == 28)
                            
                            val newCurrBody = prevRuleBody.map(b2 => renameSexpIdcs(b2, symbolMatch))
                            // println(s"newCurrBody: ${newCurrBody}")
                            val newCurrRule = SetarRule(currRule.hd, newCurrBody, currRule.filter) 
                            inlineMap(ten1) += (codeNum -> newCurrRule)
                            var finalBody = Seq[SetarExp]()
                            for ((nb, j) <- newCurrBody.zipWithIndex) { // Backtrack until that no more Inlined rules can be found
                                val (sprod, counter2, inlineMap2) = genInlinedSexp(nb, j, newPrevRule, prevLine, updatedCount, inlineMap)
                                finalBody ++= sprod
                                updatedCount += counter2
                                inlineMap ++= inlineMap2 
                            }
                            // println(s"finalBody: ${finalBody}")
                            (finalBody, updatedCount, inlineMap)
                            // } else {
                            //     (Seq(sexp), updatedCount, inlineMap)
                            // }
                        } else { // First seen, nothing to inline
                            (Seq(sexp), updatedCount, inlineMap)
                        }
                    case ten2: SetarWrapper => // Ignore tensor loading also
                        (Seq(sexp), updatedCount, inlineMap)
                }
            } else { // Ignore
                (Seq(sexp), updatedCount, inlineMap)
            }
        case t3 =>
            raise(s"Such type `${t3}` is not supported for inlining.")
    }
  }

  def inlineRule(rule: SetarRule, codeNum: Int, inlineMap: mutable.Map[SetarSym, mutable.Map[Int, SetarRule]], varsBodyInWrapperSet: mutable.Set[SetarSym]): (SetarRule, mutable.Map[SetarSym, mutable.Map[Int, SetarRule]]) = {
    var updatedInlineMap = inlineMap
    var einsumSymbolCount = 0
    var newBody = Seq[SetarExp]()
    val head = rule.hd.ten.asInstanceOf[SetarSym]
    if (updatedInlineMap.contains(head)) {
        updatedInlineMap(head) += (codeNum -> rule)
    } else {
        updatedInlineMap += (head -> mutable.Map[Int, SetarRule](codeNum -> rule))
    }
    // if (!updatedInlineMap.contains(head)) {
    //     updatedInlineMap += (head -> mutable.Map[Int, SetarRule](codeNum -> rule))
    // } 
    // val cond1 = rule.hd.idcs.size > 2
    val cond2 = !checkIfAnySymBodyInWrapper(rule, rule.hd.ten.asInstanceOf[SetarSym], varsBodyInWrapperSet)
    // val cond3 = !rule.filter.isEmpty
    // println(s"${rule.hd}: ${cond3}")
    if (cond2) {
        for ((b, i) <- rule.body.zipWithIndex) {
            // println(s"b to be inlined: ${b}")
            val (body, count, inlineMap2) = genInlinedSexp(b, i, updatedInlineMap(head)(codeNum), codeNum, einsumSymbolCount, updatedInlineMap)
            newBody ++= body
            // println(s"inlined b: ${body}\n")
            updatedInlineMap = updatedInlineMap ++ inlineMap2
            val currRest = rule.body.zipWithIndex.filter(j => j._2 > i).map(_._1)
            val tempRule = SetarRule(rule.hd, newBody ++ currRest, rule.filter) // inlined part of the rule
            updatedInlineMap(head)(codeNum) = tempRule
            einsumSymbolCount = count       
        }
        // println(s"codeNum: ${codeNum}\nrule.hd:${rule.hd}cond2:${cond2}")
        // println(s"newBody: ${newBody}\n\n")
        val newRule = SetarRule(rule.hd, newBody, rule.filter)
        updatedInlineMap(head)(codeNum) = newRule
        (newRule, updatedInlineMap)
    } else {
        (rule, updatedInlineMap)
    }
  }
}