"?p
BHostIDLE"IDLE15^?I??@A5^?I??@av?(????iv?(?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1?K7?AO@9?K7?AO@A?K7?AO@I?K7?AO@a?[b?ԙ?iU?8()????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1NbX9dN@9NbX9dN@A?(\??eM@I?(\??eM@a_??q??i?;?x???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1??C?lL@9??C?lL@A??C?lL@I??C?lL@a???????i?)?/?5???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1?A`??rC@9?A`??rC@A?A`??rC@I?A`??rC@a?",??i?:?P-????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1??? ?29@9??? ?29@A??? ?29@I??? ?29@a?XF?????iT???
???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1?Zd?4@9?Zd?4@A?Zd?4@I?Zd?4@a·?*?W??i???[\P???Unknown?
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1??????+@9??????+@A??????+@I??????+@aRE?w?i?b??~???Unknown
}	HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1?? ?r('@9?? ?r('@A?? ?r('@I?? ?r('@a?????As?iS??I????Unknown
?
HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1/?$?(@9/?$?(@Aj?t??"@Ij?t??"@aNz?օWo?i????r????Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1w??/]!@9w??/]!@Aw??/]!@Iw??/]!@a?]???l?i?0?bS????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(19??v?? @99??v?? @A9??v?? @I9??v?? @a:??~??k?ii&@??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1㥛? p @9㥛? p @A㥛? p @I㥛? p @a??wdVVk?i[???O???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1??(\?B @9??(\?B @A??(\?B @I??(\?B @a?|5?
k?iض??Y3???Unknown
iHostWriteSummary"WriteSummary(1^?I?@9^?I?@A^?I?@I^?I?@arf3?Q?i?i>??L???Unknown?
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1??????@9??????@A??????@I??????@a??61?g?i(!?9?d???Unknown
gHostStridedSlice"strided_slice(1???Mb?@9???Mb?@A???Mb?@I???Mb?@a23`p?g?i+T?b|???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1ףp=
W@9ףp=
W@Aףp=
W@Iףp=
W@aŅ????g?i???k?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1V-???@9V-???@AV-???@IV-???@a ?ғ?:g?i9?)-.????Unknown
~HostMaximum")gradient_tape/binary_crossentropy/Maximum(1?x?&1?@9?x?&1?@A?x?&1?@I?x?&1?@a?l:??f?i???>????Unknown
dHostDataset"Iterator::Model(1???K7!Q@9???K7!Q@A? ?rh?@I? ?rh?@a?5`??Be?i?M???????Unknown
lHostIteratorGetNext"IteratorGetNext(1q=
ףp@9q=
ףp@Aq=
ףp@Iq=
ףp@aa?nMn'e?i???+?????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1??ʡ?@9??ʡ?@A??ʡ?@I??ʡ?@a??? ?b?i)e???????Unknown
^HostGatherV2"GatherV2(1?V?@9?V?@A?V?@I?V?@aXr?}/b?i?XX?????Unknown
[HostAddV2"Adam/add(1\???(\@9\???(\@A\???(\@I\???(\@ao_??a?i
?r??!???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1?~j?t?@9?~j?t?@A?~j?t?@I?~j?t?@aY????:]?ii??O)0???Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1q=
ף?S@9q=
ף?S@Au?V?@Iu?V?@a?L1?P2]?i?Vx?>???Unknown
?HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1?MbX9@9?MbX9@A?MbX9@I?MbX9@a??`??\?i?5??M???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1??~j??@9??~j??@A??~j??@I??~j??@a??Y?2\?i?bd.[???Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1/?$?@9/?$?@A/?$?@I/?$?@a쒀 ??Z?iɢ4?h???Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1/?$??@9/?$??@A/?$??@I/?$??@a??NCZ?iT??ۯu???Unknown
? Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1?~j?t?
@9?~j?t?
@A?~j?t?
@I?~j?t?
@a
j͂@V?i	c?{?????Unknown
?!HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1}?5^?I
@9}?5^?I
@A}?5^?I
@I}?5^?I
@a~6????U?i$??t?????Unknown
v"HostSum"%binary_crossentropy/weighted_loss/Sum(1?rh??|	@9?rh??|	@A?rh??|	@I?rh??|	@a?Q??1U?i??GC????Unknown
`#HostGatherV2"
GatherV2_1(1???S?@9???S?@A???S?@I???S?@a???6?S?i????1????Unknown
?$HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(1B`??"?@9B`??"?@AB`??"?@IB`??"?@a?-?`>?S?i( %?????Unknown
t%Host_FusedMatMul"sequential/dense_2/BiasAdd(1D?l???@9D?l???@AD?l???@ID?l???@a?Y[_?S?i??ҝ?????Unknown
?&HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1o??ʡ@9o??ʡ??Ao??ʡ@Io??ʡ??aʒ>???R?i??!v\????Unknown
e'Host
LogicalAnd"
LogicalAnd(1?rh??|@9?rh??|@A?rh??|@I?rh??|@a?.Z	?R?it????????Unknown?
V(HostCast"Cast(1X9??v@9X9??v@AX9??v@IX9??v@a???y??R?iN???????Unknown
q)Host_FusedMatMul"sequential/dense_1/Relu(1}?5^?I@9}?5^?I@A}?5^?I@I}?5^?I@a??5_v?R?i.??,Q????Unknown
?*HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1??~j?t@9??~j?t@A??~j?t@I??~j?t@aM??Z?Q?in?	?<????Unknown
?+HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1h??|?5@9h??|?5@Ah??|?5@Ih??|?5@a??????Q?i?)?!????Unknown
?,HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1V-???@9V-???@AV-???@IV-???@a???\ʓP?i??(X????Unknown
~-HostSelect"*binary_crossentropy/logistic_loss/Select_1(1ףp=
?@9ףp=
?@Aףp=
?@Iףp=
?@a(؜??TO?i?A"E-????Unknown
v.HostNeg"%binary_crossentropy/logistic_loss/Neg(1V-??@9V-??@AV-??@IV-??@aM?O/??N?i???????Unknown
?/HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1??????@9??????@A??????@I??????@a??Y??N?i?ZD^?
???Unknown
t0HostAssignAddVariableOp"AssignAddVariableOp(1%??C?@9%??C?@A%??C?@I%??C?@a?C@???N?i?*?\???Unknown
t1HostReadVariableOp"Adam/Cast/ReadVariableOp(1-????@9-????@A-????@I-????@a??*??\M?iZu?S????Unknown
[2HostPow"
Adam/Pow_1(1d;?O?? @9d;?O?? @Ad;?O?? @Id;?O?? @a?|???K?i??M? ???Unknown
?3HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1??/?$ @9??/?$ @A??/?$ @I??/?$ @af??c+?J?iL??O'???Unknown
v4HostSub"%binary_crossentropy/logistic_loss/sub(1ˡE?????9ˡE?????AˡE?????IˡE?????ap?V?I?I?i??
??-???Unknown
?5HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1#??~j???9#??~j???A#??~j???I#??~j???a?l??G?i??j.?3???Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_2(1??|?5^??9??|?5^??A??|?5^??I??|?5^??arU`??G?i?Bܟ9???Unknown
j7HostMean"binary_crossentropy/Mean(1+??????9+??????A+??????I+??????a0?j?AG?i?T?@p????Unknown
|8HostSelect"(binary_crossentropy/logistic_loss/Select(1q=
ףp??9q=
ףp??Aq=
ףp??Iq=
ףp??a?+,?F?i???$E???Unknown
z9HostLog1p"'binary_crossentropy/logistic_loss/Log1p(15^?I??95^?I??A5^?I??I5^?I??a\7?N6uF?i??7??J???Unknown
?:HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1-??????9-??????A-??????I-??????a?MdH*F?i?w>kLP???Unknown
v;HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1???S???9???S???A???S???I???S???a;Ļ̆E?i????U???Unknown
u<HostReadVariableOp"div_no_nan/ReadVariableOp(1Zd;?O??9Zd;?O??AZd;?O??IZd;?O??a#]}?.E?i???)?Z???Unknown
?=HostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1?/?$??9?/?$??A?/?$??I?/?$??a?)^l??D?i???$`???Unknown
]>HostCast"Adam/Cast_1(1!?rh????9!?rh????A!?rh????I!?rh????ah??p?D?i?GR~Se???Unknown
o?HostReadVariableOp"Adam/ReadVariableOp(1X9??v??9X9??v??AX9??v??IX9??v??a?=G?WD?iK$iij???Unknown
?@Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1?G?z??9?G?z??A?G?z??I?G?z??aJ?i<?D?i?1s?jo???Unknown
?AHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1?p=
ף??9?p=
ף??A?p=
ף??I?p=
ף??a??͓d?B?i %??t???Unknown
`BHostDivNoNan"
div_no_nan(1????Mb??9????Mb??A????Mb??I????Mb??a?????B?iԟ??x???Unknown
oCHostMul"sequential/dropout/dropout/Mul(1??ʡE??9??ʡE??A??ʡE??I??ʡE??aZ?t?B?i???:h}???Unknown
DHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1???x?&??9???x?&??A???x?&??I???x?&??a:?)??A?i???́???Unknown
~EHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1????x???9????x???A????x???I????x???aDtc?cA?i??&????Unknown
qFHostCast"sequential/dropout/dropout/Cast(1??"??~??9??"??~??A??"??~??I??"??~??az???
A?i?`??i????Unknown
?GHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1?x?&1??9?x?&1??A?x?&1??I?x?&1??a???9?@?i>Fi??????Unknown
}HHostDivNoNan"'binary_crossentropy/weighted_loss/value(1X9??v???9X9??v???AX9??v???IX9??v???a?xwW?j@?i$?P?????Unknown
eIHostAddN"Adam/gradients/AddN(1????????9????????A????????I????????a??gDL@?i?Ca?????Unknown
?JHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1?Zd;?N@9?Zd;?N@A?K7?A`??I?K7?A`??aYa?7?@?ilȚ???Unknown
?KHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1?K7?A`??9?K7?A`??A?K7?A`??I?K7?A`??aYa?7?@?i??߫Ϟ???Unknown
?LHostDataset"NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor(1?/?$??9?/?$??A?/?$??I?/?$??a[?RN???i"??Ģ???Unknown
?MHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1?(\?????9?(\?????A?(\?????I?(\?????a?a????iRUv?????Unknown
xNHostCast"&gradient_tape/binary_crossentropy/Cast(1-??????9-??????A-??????I-??????a???ơ??i*2?땪???Unknown
?OHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1???(\???9???(\???A???(\???I???(\???a??|???>?i?a?q????Unknown
?PHostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1??"??~??9??"??~??A??"??~??I??"??~??aX2?h??>?i$st?I????Unknown
vQHostMul"%binary_crossentropy/logistic_loss/mul(15^?I??95^?I??A5^?I??I5^?I??a???I<?i??ӵ???Unknown
vRHostExp"%binary_crossentropy/logistic_loss/Exp(1`??"????9`??"????A`??"????I`??"????a?g%?b;<?i?$?Z????Unknown
rSHostAdd"!binary_crossentropy/logistic_loss(1?MbX9??9?MbX9??A?MbX9??I?MbX9??a???:?:?i?mv蹼???Unknown
THostMul".gradient_tape/sequential/dropout_1/dropout/Mul(1??C?l??9??C?l??A??C?l??I??C?l??a?M???!:?iU?o?????Unknown
?UHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1`??"????9`??"????A`??"????I`??"????a?R8??9?i_e?L6????Unknown
~VHostRealDiv")gradient_tape/binary_crossentropy/truediv(1??"??~??9??"??~??A??"??~??I??"??~??a?????[9?i?;C?a????Unknown
}WHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1u?V??9u?V??Au?V??Iu?V??aB??+)8?i????f????Unknown
vXHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1??x?&1??9??x?&1??A??x?&1??I??x?&1??a?`?E@q7?iJ??U????Unknown
vYHostAssignAddVariableOp"AssignAddVariableOp_1(1+??????9+??????A+??????I+??????a0?j?A7?i&??B=????Unknown
?ZHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1+?????9+?????A+?????I+?????a??:?7?iIR<3!????Unknown
?[HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1????x???9????x???A????x???I????x???a-&G??`6?i.?L?????Unknown
?\HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1?z?G???9?z?G???A?z?G???I?z?G???a?
??Y6?i?????????Unknown
?]HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1? ?rh???9? ?rh???A? ?rh???I? ?rh???a?5`??B5?i?h??`????Unknown
?^HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1???x?&??9???x?&??A???x?&??I???x?&??aՕO??4?i??$?????Unknown
q_HostMul" sequential/dropout/dropout/Mul_1(1J+???9J+???AJ+???IJ+???a?????4?i]-???????Unknown
?`HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1R???Q??9R???Q??AR???Q??IR???Q??a@V.94?i(?%? ????Unknown
YaHostPow"Adam/Pow(1+?????9+?????A+?????I+?????aF?h??3?i=??U?????Unknown
?bHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1?$??C??9?$??C??A?$??C??I?$??C??a=?fe9X3?i-&]????Unknown
}cHostMul",gradient_tape/sequential/dropout/dropout/Mul(1?C?l????9?C?l????A?C?l????I?C?l????a?^???G2?i?#UN????Unknown
?dHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1R???Q??9R???Q??AR???Q??IR???Q??a?ߵg??0?iآ0k????Unknown
?eHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1??/?$??9??/?$??A??/?$??I??/?$??a_[??,.?i????M????Unknown
TfHostMul"Mul(1j?t???9j?t???Aj?t???Ij?t???a??xM(?-?ip%3-????Unknown
qgHostMul" sequential/dropout_1/dropout/Mul(1???K7???9???K7???A???K7???I???K7???a??e2?)-?it????????Unknown
?hHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1?(\?????9?(\?????A?(\?????I?(\?????a???4,?i?)?????Unknown
?iHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1'1?Z??9'1?Z??A'1?Z??I'1?Z??a1??D?1+?i>?4v????Unknown
?jHostMul"0gradient_tape/sequential/dropout_1/dropout/Mul_1(1\???(\??9\???(\??A\???(\??I\???(\??a???*?i??n????Unknown
skHostCast"!sequential/dropout_1/dropout/Cast(1X9??v??9X9??v??AX9??v??IX9??v??a??r??T)?i1?
??????Unknown
lHostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1???S???9???S???A???S???I???S???a?:4?G?(?iu??a:????Unknown
?mHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1??C?l???9??C?l???A??C?l???I??C?l???a-???3'?i?ؠ??????Unknown
wnHostReadVariableOp"div_no_nan/ReadVariableOp_1(11?Zd??91?Zd??A1?Zd??I1?Zd??a|?Y??&?i?u?????Unknown
?oHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a??Ҥ??#?i?@^Y????Unknown
apHostIdentity"Identity(1?Zd;??9?Zd;??A?Zd;??I?Zd;??a.>*?iQ#?i???t?????Unknown?
sqHostMul""sequential/dropout_1/dropout/Mul_1(1m???????9m???????Am???????Im???????a???????i?S?T????Unknown
?rHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1?|?5^???9?|?5^???A?|?5^???I?|?5^???a???ռd?i?????????Unknown*?o
sHostDataset"Iterator::Model::ParallelMapV2(1?K7?AO@9?K7?AO@A?K7?AO@I?K7?AO@a1iٿ4??i1iٿ4???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1NbX9dN@9NbX9dN@A?(\??eM@I?(\??eM@ac?A????iJ??dS???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1??C?lL@9??C?lL@A??C?lL@I??C?lL@a@9G????iu???6????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1?A`??rC@9?A`??rC@A?A`??rC@I?A`??rC@ap?H9??i???/~`???Unknown
oHost_FusedMatMul"sequential/dense/Relu(1??? ?29@9??? ?29@A??? ?29@I??? ?29@a?G?l????i?|dT???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1?Zd?4@9?Zd?4@A?Zd?4@I?Zd?4@a?C?????i?$?3e????Unknown?
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1??????+@9??????+@A??????+@I??????+@a?'??c??i?KO?3???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1?? ?r('@9?? ?r('@A?? ?r('@I?? ?r('@ah?Ѥ鲕?i??u?????Unknown
?	HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1/?$?(@9/?$?(@Aj?t??"@Ij?t??"@a%9ж????i?Y,??m???Unknown
?
HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1w??/]!@9w??/]!@Aw??/]!@Iw??/]!@a|?E??i#?d?????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(19??v?? @99??v?? @A9??v?? @I9??v?? @a7??3'??i?i?l???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1㥛? p @9㥛? p @A㥛? p @I㥛? p @aQ{f~?͎?i?\?K?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1??(\?B @9??(\?B @A??(\?B @I??(\?B @a6?T??x??i!?Y??a???Unknown
iHostWriteSummary"WriteSummary(1^?I?@9^?I?@A^?I?@I^?I?@a?)?`V???i?\???????Unknown?
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1??????@9??????@A??????@I??????@a??5I̊?i???????Unknown
gHostStridedSlice"strided_slice(1???Mb?@9???Mb?@A???Mb?@I???Mb?@a0e??Ê?i?(??????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1ףp=
W@9ףp=
W@Aףp=
W@Iףp=
W@a?qhs덊?i?ʿ`:???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1V-???@9V-???@AV-???@IV-???@a? -??i?L<m?}???Unknown
~HostMaximum")gradient_tape/binary_crossentropy/Maximum(1?x?&1?@9?x?&1?@A?x?&1?@I?x?&1?@a?sp	<܈?i?b]_????Unknown
dHostDataset"Iterator::Model(1???K7!Q@9???K7!Q@A? ?rh?@I? ?rh?@a????????i??]3A???Unknown
lHostIteratorGetNext"IteratorGetNext(1q=
ףp@9q=
ףp@Aq=
ףp@Iq=
ףp@a?T ?Kև?iS???????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1??ʡ?@9??ʡ?@A??ʡ?@I??ʡ?@a6?u?sf??i?)\&????Unknown
^HostGatherV2"GatherV2(1?V?@9?V?@A?V?@I?V?@a?Z	?lO??i:O_dC???Unknown
[HostAddV2"Adam/add(1\???(\@9\???(\@A\???(\@I\???(\@a??$L???it⏐r????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1?~j?t?@9?~j?t?@A?~j?t?@I?~j?t?@adCw"?w??i??eR????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1q=
ף?S@9q=
ף?S@Au?V?@Iu?V?@al???(s??iLB	???Unknown
?HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1?MbX9@9?MbX9@A?MbX9@I?MbX9@a????#??i??? ?W???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1??~j??@9??~j??@A??~j??@I??~j??@a~jw?y??i?ٵ:????Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1/?$?@9/?$?@A/?$?@I/?$?@a~?ǂ$$~?i?h?]?????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1/?$??@9/?$??@A/?$??@I/?$??@a??4??}?i/ҩ{????Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1?~j?t?
@9?~j?t?
@A?~j?t?
@I?~j?t?
@a?"???x?ia??@???Unknown
? HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1}?5^?I
@9}?5^?I
@A}?5^?I
@I}?5^?I
@a?$?ɴ?x?i??y?q???Unknown
v!HostSum"%binary_crossentropy/weighted_loss/Sum(1?rh??|	@9?rh??|	@A?rh??|	@I?rh??|	@a?}Vb??w?i?}??????Unknown
`"HostGatherV2"
GatherV2_1(1???S?@9???S?@A???S?@I???S?@a?/I?bv?i!K????Unknown
?#HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(1B`??"?@9B`??"?@AB`??"?@IB`??"?@a?%?WZv?i.Z%??????Unknown
t$Host_FusedMatMul"sequential/dense_2/BiasAdd(1D?l???@9D?l???@AD?l???@ID?l???@a?qLvI,v?i?cX'???Unknown
?%HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1o??ʡ@9o??ʡ??Ao??ʡ@Io??ʡ??a???4u?iss??Q???Unknown
e&Host
LogicalAnd"
LogicalAnd(1?rh??|@9?rh??|@A?rh??|@I?rh??|@aӋ3u?i6??D?{???Unknown?
V'HostCast"Cast(1X9??v@9X9??v@AX9??v@IX9??v@auw[qu?i%?e'?????Unknown
q(Host_FusedMatMul"sequential/dense_1/Relu(1}?5^?I@9}?5^?I@A}?5^?I@I}?5^?I@a?;?9?t?i?*???????Unknown
?)HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1??~j?t@9??~j?t@A??~j?t@I??~j?t@a ?X?t?i)M???????Unknown
?*HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1h??|?5@9h??|?5@Ah??|?5@Ih??|?5@a?Myx*?s?i???>????Unknown
?+HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1V-???@9V-???@AV-???@IV-???@a$??r?i???XE???Unknown
~,HostSelect"*binary_crossentropy/logistic_loss/Select_1(1ףp=
?@9ףp=
?@Aףp=
?@Iףp=
?@at	F'?q?i?D?ah???Unknown
v-HostNeg"%binary_crossentropy/logistic_loss/Neg(1V-??@9V-??@AV-??@IV-??@a??)lqq?iD(?D????Unknown
?.HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1??????@9??????@A??????@I??????@ab????mq?i???????Unknown
t/HostAssignAddVariableOp"AssignAddVariableOp(1%??C?@9%??C?@A%??C?@I%??C?@a?x??&`q?i?????????Unknown
t0HostReadVariableOp"Adam/Cast/ReadVariableOp(1-????@9-????@A-????@I-????@aH?Ѭ%?p?i?7?C?????Unknown
[1HostPow"
Adam/Pow_1(1d;?O?? @9d;?O?? @Ad;?O?? @Id;?O?? @a? B??o?i?y????Unknown
?2HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1??/?$ @9??/?$ @A??/?$ @I??/?$ @aSOR?@n?i??O/???Unknown
v3HostSub"%binary_crossentropy/logistic_loss/sub(1ˡE?????9ˡE?????AˡE?????IˡE?????a???*
m?i\a@YL???Unknown
?4HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1#??~j???9#??~j???A#??~j???I#??~j???aB????j?i??Fg???Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_2(1??|?5^??9??|?5^??A??|?5^??I??|?5^??a?	ע?j?i???ځ???Unknown
j6HostMean"binary_crossentropy/Mean(1+??????9+??????A+??????I+??????aG?D#?4j?i^U?T????Unknown
|7HostSelect"(binary_crossentropy/logistic_loss/Select(1q=
ףp??9q=
ףp??Aq=
ףp??Iq=
ףp??a5??^	?i?iTF ^ŵ???Unknown
z8HostLog1p"'binary_crossentropy/logistic_loss/Log1p(15^?I??95^?I??A5^?I??I5^?I??a??	?iNi?i?O??????Unknown
?9HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1-??????9-??????A-??????I-??????a?\|y??h?i9?@?????Unknown
v:HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1???S???9???S???A???S???I???S???a%??Ah?i
?V?O ???Unknown
u;HostReadVariableOp"div_no_nan/ReadVariableOp(1Zd;?O??9Zd;?O??AZd;?O??IZd;?O??aV??˗?g?i?u"???Unknown
?<HostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1?/?$??9?/?$??A?/?$??I?/?$??a??J??rg?i??Ȟy/???Unknown
]=HostCast"Adam/Cast_1(1!?rh????9!?rh????A!?rh????I!?rh????aw?ޙ{[g?i??b?F???Unknown
o>HostReadVariableOp"Adam/ReadVariableOp(1X9??v??9X9??v??AX9??v??IX9??v??a???.?f?i?r@I?]???Unknown
??Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1?G?z??9?G?z??A?G?z??I?G?z??a??!??f?it??[Qt???Unknown
?@HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1?p=
ף??9?p=
ף??A?p=
ף??I?p=
ף??a????6e?i?]??????Unknown
`AHostDivNoNan"
div_no_nan(1????Mb??9????Mb??A????Mb??I????Mb??a?3??@?d?i??E?????Unknown
oBHostMul"sequential/dropout/dropout/Mul(1??ʡE??9??ʡE??A??ʡE??I??ʡE??a??)Bc?d?i?/??_????Unknown
CHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1???x?&??9???x?&??A???x?&??I???x?&??a:???c?i?i?d1????Unknown
~DHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1????x???9????x???A????x???I????x???aeR+*?c?i?????????Unknown
qEHostCast"sequential/dropout/dropout/Cast(1??"??~??9??"??~??A??"??~??I??"??~??a^?U?`4c?i?????????Unknown
?FHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1?x?&1??9?x?&1??A?x?&1??I?x?&1??a?gJ ?b?iS5?? ???Unknown
}GHostDivNoNan"'binary_crossentropy/weighted_loss/value(1X9??v???9X9??v???AX9??v???IX9??v???a5s??b?i?:?C???Unknown
eHHostAddN"Adam/gradients/AddN(1????????9????????A????????I????????a?bht]b?i??w?%???Unknown
?IHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1?Zd;?N@9?Zd;?N@A?K7?A`??I?K7?A`??a?:fK?'b?i?`0?7???Unknown
?JHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1?K7?A`??9?K7?A`??A?K7?A`??I?K7?A`??a?:fK?'b?i5j???I???Unknown
?KHostDataset"NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor(1?/?$??9?/?$??A?/?$??I?/?$??a3?J?a?iEC?3?[???Unknown
?LHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1?(\?????9?(\?????A?(\?????I?(\?????a?ِ??a?i??#?m???Unknown
xMHostCast"&gradient_tape/binary_crossentropy/Cast(1-??????9-??????A-??????I-??????a??9n{a?i?M(???Unknown
?NHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1???(\???9???(\???A???(\???I???(\???ao??a?ca?i}ۮ%f????Unknown
?OHostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1??"??~??9??"??~??A??"??~??I??"??~??a P?Y?Ta?i?`ɺ????Unknown
vPHostMul"%binary_crossentropy/logistic_loss/mul(15^?I??95^?I??A5^?I??I5^?I??a???1l?_?i.W!?????Unknown
vQHostExp"%binary_crossentropy/logistic_loss/Exp(1`??"????9`??"????A`??"????I`??"????ao??)?_?it)6??????Unknown
rRHostAdd"!binary_crossentropy/logistic_loss(1?MbX9??9?MbX9??A?MbX9??I?MbX9??a?gNg^?i<?i/?????Unknown
SHostMul".gradient_tape/sequential/dropout_1/dropout/Mul(1??C?l??9??C?l??A??C?l??I??C?l??a+???q]?iR?Z????Unknown
?THostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1`??"????9`??"????A`??"????I`??"????aﮆ?7]?i?./"????Unknown
~UHostRealDiv")gradient_tape/binary_crossentropy/truediv(1??"??~??9??"??~??A??"??~??I??"??~??a?ij?\?i5cd?K????Unknown
}VHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1u?V??9u?V??Au?V??Iu?V??a?Q??9[?i^k???	???Unknown
vWHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1??x?&1??9??x?&1??A??x?&1??I??x?&1??a_tA@kjZ?i?\????Unknown
vXHostAssignAddVariableOp"AssignAddVariableOp_1(1+??????9+??????A+??????I+??????aG?D#?4Z?is.n8$???Unknown
?YHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1+?????9+?????A+?????I+?????a?.?OZ?i?v?6?1???Unknown
?ZHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1????x???9????x???A????x???I????x???a"6??b7Y?i%?B??=???Unknown
?[HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1?z?G???9?z?G???A?z?G???I?z?G???a?y??/Y?i??rJ???Unknown
?\HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1? ?rh???9? ?rh???A? ?rh???I? ?rh???a??????W?iZCmV???Unknown
?]HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1???x?&??9???x?&??A???x?&??I???x?&??a?`۶6?W?i??_?5b???Unknown
q^HostMul" sequential/dropout/dropout/Mul_1(1J+???9J+???AJ+???IJ+???a?*??܁W?iX???m???Unknown
?_HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1R???Q??9R???Q??AR???Q??IR???Q??a??0K??V?i??ܞ[y???Unknown
Y`HostPow"Adam/Pow(1+?????9+?????A+?????I+?????a???NV?i!??????Unknown
?aHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1?$??C??9?$??C??A?$??C??I?$??C??a[??V?U?i?eB4i????Unknown
}bHostMul",gradient_tape/sequential/dropout/dropout/Mul(1?C?l????9?C?l????A?C?l????I?C?l????a??N?T?i?P۵????Unknown
?cHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1R???Q??9R???Q??AR???Q??IR???Q??a\?E)
S?iʟ??:????Unknown
?dHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1??/?$??9??/?$??A??/?$??I??/?$??ah%?+4 Q?iݛ	
?????Unknown
TeHostMul"Mul(1j?t???9j?t???Aj?t???Ij?t???aɸg??P?i?O?+????Unknown
qfHostMul" sequential/dropout_1/dropout/Mul(1???K7???9???K7???A???K7???I???K7???as!J?\nP?i????b????Unknown
?gHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1?(\?????9?(\?????A?(\?????I?(\?????aHq?%e?O?i?T?U????Unknown
?hHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1'1?Z??9'1?Z??A'1?Z??I'1?Z??a[i$???N?i ^q??????Unknown
?iHostMul"0gradient_tape/sequential/dropout_1/dropout/Mul_1(1\???(\??9\???(\??A\???(\??I\???(\??a??7?SbM?i??g?V????Unknown
sjHostCast"!sequential/dropout_1/dropout/Cast(1X9??v??9X9??v??AX9??v??IX9??v??ap?Dfg?L?i<}A?y????Unknown
kHostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1???S???9???S???A???S???I???S???a??=L?il??y????Unknown
?lHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1??C?l???9??C?l???A??C?l???I??C?l???a??V%J?i!+S????Unknown
wmHostReadVariableOp"div_no_nan/ReadVariableOp_1(11?Zd??91?Zd??A1?Zd??I1?Zd??ayͺ؅?I?i?Y??m????Unknown
?nHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a???!?|F?i??-????Unknown
aoHostIdentity"Identity(1?Zd;??9?Zd;??A?Zd;??I?Zd;??a3e???E?i?d}W~????Unknown?
spHostMul""sequential/dropout_1/dropout/Mul_1(1m???????9m???????Am???????Im???????aS?r??;?iU?ߓ?????Unknown
?qHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1?|?5^???9?|?5^???A?|?5^???I?|?5^???a]Iea8?i?????????Unknown2CPU