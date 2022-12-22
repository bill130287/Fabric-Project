package system

var LayersName = [3]string{"L1.weight", "L1.bias", "L2.weight"}
var LayersShape = [3]string{"10", "5", "3"}
var LayersSize = [3]uint64{10, 5, 3}
var LayersParameter = map[string][]string{
	"L1.weight": {"24104972945455764697340839814202094568375648879789887726390147154385647784189", "32350226411570111508078993588387803843521409181086397968019868449307760298768", "17668826364676433291318543647183916229452783623142128050746221096791914648117", "6302489740713797966299008496418990484771181892052942252359961679568608216971", "27681830945091345544083499999529500190076513222903841811341897244206114357156", "15573903626917250350654494669951376838197754774086072561600289773386141850380", "5570536266464237447858137469449416393266345957993564864953902181550457262387", "33115309482021590584041909726456229175171631334184060037748614736608500698817", "31881344855654959843982020641744757686127379180641814223627675851255663066745", "4976297491134355222161424592635849311581086572875842679084215999845749076066"},
	"L1.bias":   {"10361820327773971660697166259248465148718712076118051112511742004063939085", "1505552383649379991855787233750982030705179485299948483054489954919609063380", "29500358773155911526065724500183699455918976983009307953791289089739208476347", "12051337663166654784505379621727684061942176062844665046398432053886166228555", "17430266092638082666145628005798051442891246169968753898065988110975406522615"},
	"L2.weight": {"27621905015028971999511940450076427819856336014615967804659273771369721836408", "32940991006929342492906091326626548188962833008001654853640299384170423206367", "30013865525516555398983908521061379739561327413627155656198081055148215797966"},
}