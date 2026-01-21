// Optional evaluation DLL for NullMove.
// Build on Windows with MSVC cl.exe (see scripts/build_cpp_eval.ps1).

#include <cstdint>

extern "C" __declspec(dllexport) int nm_eval(
    uint64_t wp, uint64_t wn, uint64_t wb, uint64_t wr, uint64_t wq, uint64_t wk,
    uint64_t bp, uint64_t bn, uint64_t bb, uint64_t br, uint64_t bq, uint64_t bk,
    int turn_white
) {
    // Piece values (centipawns)
    constexpr int V_P = 100;
    constexpr int V_N = 320;
    constexpr int V_B = 330;
    constexpr int V_R = 500;
    constexpr int V_Q = 900;

    // PSTs from eval.py (white perspective). Index = square 0..63 (a1=0).
    static const int PST_P[64] = {
        0,0,0,0,0,0,0,0,
        5,10,10,-20,-20,10,10,5,
        5,-5,-10,0,0,-10,-5,5,
        0,0,0,20,20,0,0,0,
        5,5,10,25,25,10,5,5,
        10,10,20,30,30,20,10,10,
        50,50,50,50,50,50,50,50,
        0,0,0,0,0,0,0,0
    };
    static const int PST_N[64] = {
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,0,5,5,0,-20,-40,
        -30,5,10,15,15,10,5,-30,
        -30,0,15,20,20,15,0,-30,
        -30,5,15,20,20,15,5,-30,
        -30,0,10,15,15,10,0,-30,
        -40,-20,0,0,0,0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    };
    static const int PST_B[64] = {
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,5,0,0,0,0,5,-10,
        -10,10,10,10,10,10,10,-10,
        -10,0,10,10,10,10,0,-10,
        -10,5,5,10,10,5,5,-10,
        -10,0,5,10,10,5,0,-10,
        -10,0,0,0,0,0,0,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    };
    static const int PST_R[64] = {
        0,0,0,5,5,0,0,0,
        -5,0,0,0,0,0,0,-5,
        -5,0,0,0,0,0,0,-5,
        -5,0,0,0,0,0,0,-5,
        -5,0,0,0,0,0,0,-5,
        -5,0,0,0,0,0,0,-5,
        5,10,10,10,10,10,10,5,
        0,0,0,0,0,0,0,0
    };
    static const int PST_Q[64] = {
        -20,-10,-10,-5,-5,-10,-10,-20,
        -10,0,0,0,0,0,0,-10,
        -10,0,5,5,5,5,0,-10,
        -5,0,5,5,5,5,0,-5,
        0,0,5,5,5,5,0,-5,
        -10,5,5,5,5,5,0,-10,
        -10,0,5,0,0,0,0,-10,
        -20,-10,-10,-5,-5,-10,-10,-20
    };
    static const int PST_K[64] = {
        20,30,10,0,0,10,30,20,
        20,20,0,0,0,0,20,20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30
    };

    auto pop_lsb = [](uint64_t &bb) -> int {
        unsigned long idx;
        _BitScanForward64(&idx, bb);
        bb &= (bb - 1);
        return static_cast<int>(idx);
    };

    auto mirror_sq = [](int sq) -> int {
        // mirror along horizontal axis (rank flip): sq ^ 56
        return sq ^ 56;
    };

    int score_white = 0;

    auto add_piece = [&](uint64_t bb, int value, const int pst[64], bool is_white) {
        while (bb) {
            int sq = pop_lsb(bb);
            score_white += is_white ? value : -value;
            int psq = is_white ? pst[sq] : -pst[mirror_sq(sq)];
            score_white += psq;
        }
    };

    add_piece(wp, V_P, PST_P, true);
    add_piece(wn, V_N, PST_N, true);
    add_piece(wb, V_B, PST_B, true);
    add_piece(wr, V_R, PST_R, true);
    add_piece(wq, V_Q, PST_Q, true);
    add_piece(wk, 0,   PST_K, true);

    add_piece(bp, V_P, PST_P, false);
    add_piece(bn, V_N, PST_N, false);
    add_piece(bb, V_B, PST_B, false);
    add_piece(br, V_R, PST_R, false);
    add_piece(bq, V_Q, PST_Q, false);
    add_piece(bk, 0,   PST_K, false);

    // Return side-to-move perspective.
    return turn_white ? score_white : -score_white;
}
