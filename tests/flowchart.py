import graphviz

dot_code = """
digraph G {
    graph [dpi=600, pad=0.1, rankdir=TB, splines=ortho, ranksep=0.5, nodesep=0.5];
    node [shape=box, width=2.25, fontname = "DejaVu Sans"];
    edge [arrowsize=0.7];
    A [label=<
        <TABLE BORDER="0" CELLBORDER="0" CELLPADDING="0" ALIGN="LEFT">
            <TR><TD ALIGN="LEFT"><B>Raw Data</B></TD></TR>
            <TR><TD ALIGN="LEFT">• Recordings</TD></TR>
            <TR><TD ALIGN="LEFT">• YAML config</TD></TR>
        </TABLE>
    >];

    B [label=<
        <TABLE BORDER="0" CELLBORDER="0" CELLPADDING="0" ALIGN="LEFT">
            <TR><TD ALIGN="LEFT"><B>Load and epoch</B></TD></TR>
            <TR><TD ALIGN="LEFT">• MNE-BIDS / Neo</TD></TR>
            <TR><TD ALIGN="LEFT">• Segments / continuous</TD></TR>
        </TABLE>
    >];

    C [label=<
        <TABLE BORDER="0" CELLBORDER="0" CELLPADDING="0" ALIGN="LEFT">
            <TR><TD ALIGN="LEFT"><B>Preprocess</B></TD></TR>
            <TR><TD ALIGN="LEFT">• Baseline</TD></TR>
            <TR><TD ALIGN="LEFT">• Remove artifacts</TD></TR>
            <TR><TD ALIGN="LEFT">• Average</TD></TR>
            <TR><TD ALIGN="LEFT">• Smooth</TD></TR>
        </TABLE>
    >];

    D [label=<
        <TABLE BORDER="0" CELLBORDER="0" CELLPADDING="0" ALIGN="LEFT">
            <TR><TD ALIGN="LEFT"><B>Template match</B></TD></TR>
            <TR><TD ALIGN="LEFT">• Matched filter</TD></TR>
            <TR><TD ALIGN="LEFT">• GLRT</TD></TR>
        </TABLE>
    >];

    E [label=<
        <TABLE BORDER="0" CELLBORDER="0" CELLPADDING="0" ALIGN="LEFT">
            <TR><TD ALIGN="LEFT"><B>Quantify</B></TD></TR>
            <TR><TD ALIGN="LEFT">• Amplitude</TD></TR>
            <TR><TD ALIGN="LEFT">• Slope</TD></TR>
            <TR><TD ALIGN="LEFT">• Latency</TD></TR>
        </TABLE>
    >];

    F [label=<
        <TABLE BORDER="0" CELLBORDER="0" CELLPADDING="0" ALIGN="LEFT">
            <TR><TD ALIGN="LEFT"><B>Summarize</B></TD></TR>
            <TR><TD ALIGN="LEFT">• Plots</TD></TR>
            <TR><TD ALIGN="LEFT">• Excel / YAML</TD></TR>
        </TABLE>
    >];

    A -> B -> C -> D -> E -> F;
    # A -> B -> C;
    # C -> D;
    # D -> E;
    # E -> F;

    # {rank=same; A; B; C;}
    # {rank=same; F; E; D;}

    # A -> F [style=invis, weight=10];
    # B -> E [style=invis, weight=10];
    }
"""

src = graphviz.Source(dot_code)

src.render('flowchart_fig1_new1', format='svg', view=True)