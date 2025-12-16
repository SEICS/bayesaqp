package bayestensor
package utils

import scala.sys.process._
import java.net._
import java.io._
import scala.io._

object PythonProxy {
  lazy val (address, port) = {
    val jsStr = scala.io.Source.fromFile("server/conf.json").mkString
    val js = ujson.read(jsStr)
    (js("address").str -> js("port").num.toInt)
  }
  def runPythonCode(pythonCode: String): String = {
    val pr = Process(Seq("python3", "-c", pythonCode))
    // println(s"pr: ${pr}")
    // println(s"pr.!!: ${pr.!!}")
    pr.!!
  }

  def runPythonCodeServer(pythonCode: String): String = {
    val sock = new Socket(InetAddress.getByName(address), port)
    val in = new BufferedSource(sock.getInputStream()).getLines()
    val out = new PrintStream(sock.getOutputStream())
    val msg = pythonCode + "    <END>"
    out.print(msg)
    out.flush()
    val result = {
      var res = ""
      while(in.hasNext) {
        res += in.next()
      }
      res
    }
    // println("Received: " + result)
    sock.close()
    result
  }

}