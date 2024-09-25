lazy val root = (project in file("."))
  .settings(
    organization := "com.example",
    name := "example",
    version := "0.0.1-SNAPSHOT",
    scalaVersion := "2.13.10",
    libraryDependencies ++= Seq(
      "org.clulab" % "timenorm_2.13" % "1.0.5",
      "net.sf.py4j" % "py4j" % "0.10.7"
    ),
    addCompilerPlugin("org.typelevel" %% "kind-projector"     % "0.13.2" cross CrossVersion.full),
    addCompilerPlugin("com.olegpy"    %% "better-monadic-for" % "0.3.1"),
    testFrameworks += new TestFramework("munit.Framework")
  )
