package com.example

import org.clulab.timenorm.scfg._
import scala.util.Success
//import scala.collection.mutable.HashMap

class TimeNormer {
  val parser = TemporalExpressionParser.en()
  def norm(timex: java.lang.String, anchorDate: java.util.HashMap[String, Int]): String = {
    //val (y, m, d) = anchorDate
    val Success(temporal) = parser.parse(timex, TimeSpan.of(anchorDate.get("y"), anchorDate.get("m"), anchorDate.get("d")))
    return temporal.toString()
  }
}
