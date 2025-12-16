val scala_version = "2.12.11"

lazy val bayestensor = (project in file("."))
  .enablePlugins(NativeImagePlugin)
  .settings(
    scalaVersion := scala_version,
    Compile / mainClass := Some("bayestensor.driver.Driver"),
    nativeImageGraalHome := java.nio.file.Paths.get(sys.env.getOrElse("HOME", ".") + "/.sdkman/candidates/java/21.0.2-graalce"),
    libraryDependencies ++= Seq(
      "com.lihaoyi" %% "pprint"    % "0.7.0",
      "com.lihaoyi" %% "fastparse" % "3.1.1",
      "com.lihaoyi" %% "upickle"   % "3.1.0",
      "com.lihaoyi" %% "os-lib"    % "0.9.1",
      "junit"        % "junit-dep" % "4.10" % Test,
      "org.scalatest" %% "scalatest" % "3.2.18" % Test
    )
  )