module logic_circuit (
    input wire A, B, C, D,
    output wire Y
);
    wire xnor_ab, nor_c, or_out;
    xnor(xnor_ab, A, B);
    nor(nor_c, xnor_ab, C);
    or(Y, nor_c, D);
endmodule
