
import { dir } from "console";
import FabricClient from "./FabricClient"

(async () => {
  console.log('start')
  const fabricClient = new FabricClient()

  await fabricClient.connectNetwork()

  //* fabric_proj *//
  const N = "5" // 參與FL的client數
  const dataset = "/MNIST"
  var main_dir = "/home/tsengpo/fabric_proj/local/ExperimentFile/2/in_clients_" + N + dataset // + "/budget_4e-06" // "/home/tsengpo/fabric_proj/Client/ExperimentFile/1/in_clients_" + N + dataset
  var round = "/round1"

  // Init 
  var pk1 = "58801821942098389655608637268571891952766735145343720439700155518909357497939"
  var pk2 = "59186167506438882094954891140012136117015581455262837122876539691233661307113"
  var initparams = require(main_dir + "/initparams.json")
  // 測試只上傳parameters: const initResult = await fabricClient.invokeChaincode('Init', [JSON.stringify(initparams.parameters), pk1, pk2, N])
  var initnoise = require(main_dir + "/initnoise.json")
  // console.log(initparams.parameters)
  // console.log(initnoise.parameters)
  const initResult = await fabricClient.invokeChaincode('Init', [JSON.stringify(initparams.parameters), JSON.stringify(initnoise.parameters), pk1, pk2, N])
  console.log('init result', '\n', initResult)

  /* Upload and Update 分開 上傳1 round的資料, 要測下1 round要更改上面round變數(資料夾位置)*/
  // for (var i=0; i<Number(N); i++) {
  //   var client_file = "/P" + i.toString(10) + "_"
  //   console.log(main_dir + round + client_file)
  //   var gradsJson = require(main_dir + round + client_file + "grads.json")
  //   var noiseJson = require(main_dir + round + client_file + "noise.json")

  //   var before = Date.now()
  //   await fabricClient.invokeChaincode('OnlyUpload', [JSON.stringify(gradsJson.grads), JSON.stringify(noiseJson.grads)])
  //   var after = Date.now()
  //   console.log("upload time cost: ", after - before)
  //   if (i == Number(N) - 1) {
  //     var before = Date.now()
  //     await fabricClient.invokeChaincode('OnlyUpdate', [])
  //     var after = Date.now()
  //     console.log("update time cost: ", after - before)  
  //   }
  // }  

  /* 不用for loop測1 round 只單獨測某一個client的時間*/
  // var client_file = "/P4_"
  // console.log(main_dir + round + client_file)
  // var gradsJson = require(main_dir + round + client_file + "grads.json")
  // var noiseJson = require(main_dir + round + client_file + "noise.json")

  // var before = Date.now()
  // await fabricClient.invokeChaincode('OnlyUpload', [JSON.stringify(gradsJson.grads), JSON.stringify(noiseJson.grads)])
  // var after = Date.now()
  // console.log("upload time cost: ", after - before)

  // var before = Date.now()
  // await fabricClient.invokeChaincode('OnlyUpdate', [])
  // var after = Date.now()
  // console.log("update time cost: ", after - before)  

  /*每次跑 1 round (含Upload, Download params, Download noises), 跑完 1 round 更改上面round變數 讀取其他round的資料*/
  // for (var i=0; i<Number(N); i++) {
  //   var client_file = "/P" + i.toString(10) + "_"
  //   console.log(main_dir + round + client_file)
  //   var gradsJson = require(main_dir + round + client_file + "grads.json")
  //   var noiseJson = require(main_dir + round + client_file + "noise.json")

  //   var before = Date.now()
  //   await fabricClient.invokeChaincode('Upload2', [JSON.stringify(gradsJson.grads), JSON.stringify(noiseJson.grads)])
  //   var after = Date.now()
  //   console.log("upload time cost: ", after - before)
  // }

  // // Download params
  // var before = Date.now()
  // const downloadResult = await fabricClient.queryChaincode('DownloadParams', [])
  // var after = Date.now()
  // console.log("download param time cost: ", after - before)
  // // console.log('download result', '\n', downloadResult)
  // console.log('successfully download params')
  // const fs = require('fs')
  // fs.writeFileSync(main_dir + round + "/download_params.json", JSON.stringify(downloadResult)) //最新

  // // Download noises
  // before = Date.now()
  // const downloadNoise = await fabricClient.queryChaincode('DownloadNoise', [])
  // after = Date.now()
  // console.log("download noise time cost: ", after - before)
  // // console.log('download noise', '\n', downloadNoise)
  // console.log('sucessfully download noises')
  // const noisefs = require('fs')
  // noisefs.writeFileSync(main_dir + round + "/download_noises.json", JSON.stringify(downloadNoise))


  /* fabric_proj new 可以連續執行 一次上傳好幾個round資料 調整 i(起點round) 和 total_rounds, 建議不要一次太多*/
  // const N = "5"
  // const dataset = "/MNIST"
  // var main_dir = "/home/tsengpo/fabric_proj/local/ExperimentFile/2/in_clients_" + N + dataset

  // Init 
  // var pk1 = "83142840061319176461984013641318012358493371133127224302603253214248687462249"
  // var pk2 = "70622668753950473530241617878165646370478272191663550448368747512320705655537"
  // var initparams = require(main_dir + "/initparams.json")
  // // 測試只上傳parameters: const initResult = await fabricClient.invokeChaincode('Init', [JSON.stringify(initparams.parameters), pk1, pk2, N])
  // var initnoise = require(main_dir + "/initnoise.json")
  // // console.log(initparams.parameters)
  // // console.log(initnoise.parameters)
  // const initResult = await fabricClient.invokeChaincode('Init', [JSON.stringify(initparams.parameters), JSON.stringify(initnoise.parameters), pk1, pk2, N])
  
  
  // Upload and download
  // const total_rounds = 50

  // for (var i=1; i<=total_rounds; i++) {
  //   var round_dir = "/round" + i.toString(10)
  //   for (var j=0; j<Number(N); j++) {
  //     var client_file = "/P" + j.toString(10) + "_"
  //     console.log(main_dir + round_dir + client_file)
  //     var gradsJson = require(main_dir + round_dir + client_file + "grads.json")
  //     var noiseJson = require(main_dir + round_dir + client_file + "noise.json")
  //     await fabricClient.invokeChaincode('Upload2', [JSON.stringify(gradsJson.grads), JSON.stringify(noiseJson.grads)])
  //   }

  //   const downloadResult = await fabricClient.queryChaincode('DownloadParams', [])
  //   console.log('successfully download params')
  //   const fs = require('fs')
  //   fs.writeFileSync(main_dir + round_dir + "/download_params.json", JSON.stringify(downloadResult))

  //   const downloadNoise = await fabricClient.queryChaincode('DownloadNoise', [])
  //   console.log('sucessfully download noises')
  //   const noisefs = require('fs')
  //   noisefs.writeFileSync(main_dir + round_dir + "/download_noises.json", JSON.stringify(downloadNoise))
  // }


  console.log('closeNetwork')
  fabricClient.closeNetwork()
})();

// function sleep(ms: any) {
//   return new Promise((resolve) => {
//     setTimeout(resolve, ms);
//   });
// }