import sys

sys.path.append("../")

import astroquery.vizier
import pandas as pd
import numpy as np

"""
リングのaugmentationパターンを作成するスクリプト
"""


def catalogue(choice, ring_select):
    if choice == "CH":
        viz = astroquery.vizier.Vizier(columns=["*"])
        viz.ROW_LIMIT = -1
        bub_2006 = viz.query_constraints(catalog="J/ApJ/649/759/bubbles")[0].to_pandas()
        bub_2007 = viz.query_constraints(catalog="J/ApJ/670/428/bubble")[0].to_pandas()
        bub_2006_change = bub_2006.set_index("__CPA2006_")
        bub_2007_change = bub_2007.set_index("__CWP2007_")
        CH = pd.concat([bub_2006_change, bub_2007_change])
        CH["CH"] = CH.index
        if ring_select:
            print("#######################")
            print("   Ring selection")
            print("#######################")
            rank_2_3 = np.load("rank_3.npy")
            CH = CH.loc[rank_2_3]
        return CH

    elif choice == "MWP":
        viz = astroquery.vizier.Vizier(columns=["*"])
        viz.ROW_LIMIT = -1
        MWP = viz.query_constraints(catalog="2019yCat..74881141J ")[0].to_pandas()
        MWP.loc[MWP["GLON"] >= 358.446500015535, "GLON"] -= 360
        MWP = MWP.set_index("MWP")
        MWP["MWP"] = MWP.index
        return MWP

    else:
        print("this choice catalogue does not exist")
