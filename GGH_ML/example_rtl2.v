module logic_circuit (
    input A, B, C, D, E,
    output Y
);
    wire nand_out, or_out, and1_out;
    nand (nand_out, A, B);
    or (or_out, nand_out, C);
    and (and1_out, or_out, D);
    and (Y, and1_out, E);
endmodule
