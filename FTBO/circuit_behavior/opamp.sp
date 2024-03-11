.subckt opamp in vo gnd
G_in_1 1 gnd in gnd G_in_1
Rprs_in_1 1 gnd 'Rprs_in_1/G_in_1'
G_1_2 2 gnd 1 gnd '-1*G_1_2'
Rprs_1_2 2 gnd 'Rprs_1_2/G_1_2'
Cprsin_1_2 1 gnd 'G_1_2/6.28*5n'
G_2_vo vo gnd 2 gnd G_2_vo
Rprs_2_vo vo gnd 'Rprs_2_vo/G_2_vo'
Cprsin_2_vo 2 gnd 'G_2_vo/6.28*5n'
G_in_2_ff 2 gnd in gnd G_in_2_ff
Rprs_in_2_ff 2 gnd 'Rprs_in_2_ff/G_in_2_ff'
Cprsout_in_2_ff 2 gnd Cprsout_in_2_ff
G_in_vo_ff vo gnd in gnd '-1*G_in_vo_ff'
Rprs_in_vo_ff vo gnd 'Rprs_in_vo_ff/G_in_vo_ff'
Cprsout_in_vo_ff vo gnd Cprsout_in_vo_ff
R_1_1_vo 1 1_vo R_1_1_vo
C_1_vo_vo 1_vo vo C_1_vo_vo
R_2_2_vo 2 2_vo R_2_2_vo
C_2_vo_vo 2_vo vo C_2_vo_vo
R_1_1_gnd 1 1_gnd R_1_1_gnd
C_1_gnd_gnd 1_gnd gnd C_1_gnd_gnd
R_L vo gnd r=0.15meg
C_L vo gnd c=10n
.ends