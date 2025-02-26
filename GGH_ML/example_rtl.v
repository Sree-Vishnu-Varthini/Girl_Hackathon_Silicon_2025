module logic_circuit (
    input wire A, B, D,
    output wire Y
);
    wire xor_out, nor_out;

    xor  (xor_out, A, D);                     nor  (nor_out, xor_out, B);          nand (Y, nor_out, A);             
endmodule