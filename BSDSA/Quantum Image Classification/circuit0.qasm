OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg meas[4];
h q[0];
h q[1];
h q[2];
h q[3];
ry(2.0182894550320896) q[0];
ry(0.7530566276470224) q[1];
ry(1.9475124758841351) q[2];
ry(0.7590649829994611) q[3];
rx(1.9278687495276803) q[0];
rx(1.7361788135413279) q[1];
rx(0.4941018386314762) q[2];
rx(1.0875772349825594) q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[0];
rx(0.10259388951907539) q[0];
rx(0.08010437522560485) q[1];
rx(1.9098126648299862) q[2];
rx(1.6758070443170086) q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[0];
rx(1.5030486979282176) q[0];
rx(0.04241758659371511) q[1];
rx(0.16474311564349858) q[2];
rx(0.32518332202674705) q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[0];
rx(0.42708746249325996) q[0];
rx(0.3880777511455368) q[1];
rx(0.2904071126784137) q[2];
rx(0.3602748932218497) q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[0];
rx(0.06281352221603945) q[0];
rx(0.30212241364911924) q[1];
rx(0.7607850486168974) q[2];
rx(0.5208665342284463) q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[0];
rx(0.770967179954561) q[0];
rx(0.4480841515665673) q[1];
rx(0.48868310378907237) q[2];
rx(0.25607632876562253) q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[0];
barrier q[0],q[1],q[2],q[3];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];