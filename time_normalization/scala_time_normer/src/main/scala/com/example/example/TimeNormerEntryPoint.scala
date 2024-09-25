package com.example

import py4j.GatewayServer

class TimeNormEntryPoint {
  var timeNormer: TimeNormer = new TimeNormer()
  
  def getTimeNormer(): TimeNormer = timeNormer
}

object TimeNormEntryPoint extends App {
  val gatewayServer = new GatewayServer(new TimeNormEntryPoint())
  gatewayServer.start()
  println("Gateway server started")
  
  while (true) {
    Thread.sleep(Long.MaxValue)
  }
}
