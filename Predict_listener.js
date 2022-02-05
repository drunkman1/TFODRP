// temperature-listener.js
var mc = require('mcprotocol');
var conn = new mc;
var doneReading = false;
var doneWriting = false;
var variables = { TEST1: 'D0,5', 	// 5 words starting at D0
	  TEST2: 'M6990,28', 			// 28 bits at M6990
	  TEST3: 'CN199,2',			// ILLEGAL as CN199 is 16-bit, CN200 is 32-bit, must request separately
	  TEST4: 'R2000,2',			// 2 words at R2000
	  TEST5: 'X034',				// Simple input
	  TEST6: 'D6000.1,20',			// 20 bits starting at D6000.1
	  TEST7: 'D6001.2',				// Single bit at D6001
	  TEST8: 'S4,2',				// 2 bits at S4
	  TEST9: 'RFLOAT5000,40'		// 40 floating point numbers at R5000	
};										// See setTranslationCB below for more examples

//	conn.addItems(['TEST1', 'TEST4']);	
//	conn.addItems('TEST6');
//	conn.removeItems(['TEST2', 'TEST3']);  // We could do this.  
//  conn.writeItems(['TEST5', 'TEST7'], [ true, true ], valuesWritten);  	// You can write an array of items as well.  
//	conn.writeItems('TEST4', [ 666, 777 ], valuesWritten);  				// You can write a single array item too.  
//	conn.readAllItems(valuesReady);
//  conn.writeItems('TEST5', true, valuesWritten)


const { spawn } = require('child_process');
// const predicts = []; // Store readings
const sensor = spawn('python', ['OD_Predict.py']);
conn.initiateConnection({port: 1281, host: '192.168.0.2', ascii: false}, connected);

function connected(err){
    while(typeof(err) == "undefined"){
        sensor.stdout.on('data', function(data) {
            const predicts = []   // store reading
            conn.setTranslationCB(function(tag) {return variables[tag];});
             
            // Coerce Buffer object to Float
            // predicts.push(parseFloat(data));
			predicts.push(parseInt(data));
            console.log(predicts);
            if (predicts == 1){conn.writeItems('TEST5', true, valuesWritten)}
            if (predicts == 0){conn.writeItems('TEST5', false, valuesWritten)} 
        
        });
    }
}

function valuesReady(anythingBad, values) {
	if (anythingBad) { console.log("SOMETHING WENT WRONG READING VALUES!!!!"); }
	console.log(values);
	doneReading = true;
	if (doneWriting) { process.exit(); }
}

function valuesWritten(anythingBad) {
	if (anythingBad) { console.log("SOMETHING WENT WRONG WRITING VALUES!!!!"); }
	console.log("Done writing.");
	doneWriting = true;
	// if (doneReading) { process.exit(); }
}
