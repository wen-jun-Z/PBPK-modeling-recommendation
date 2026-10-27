"""Hand-authored SVG medical illustrations for the blueprint workflow.

These illustrations use smooth anatomical paths and gradients.  They do not
read images, trace pixels, import a vectorized-image cache, or embed bitmaps.
Coordinates match the selected reference's 1605 × 980 layout.
"""

from xml.etree import ElementTree as ET

NS = "http://www.w3.org/2000/svg"


def _human() -> ET.Element:
    return ET.fromstring(r'''
    <g xmlns="http://www.w3.org/2000/svg" id="native_human"
       transform="translate(38 83) scale(.95 .95)">
      <title>Human physiology template: circulation, thoracic and abdominal organs</title>
      <defs>
        <linearGradient id="nha_skin" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stop-color="#799bbf" stop-opacity=".75"/>
          <stop offset=".18" stop-color="#e1e7f1" stop-opacity=".92"/>
          <stop offset=".5" stop-color="#eee9ec" stop-opacity=".93"/>
          <stop offset=".82" stop-color="#dce5f2" stop-opacity=".92"/>
          <stop offset="1" stop-color="#799fc5" stop-opacity=".75"/>
        </linearGradient>
        <radialGradient id="nha_face" cx="45%" cy="33%" r="75%">
          <stop stop-color="#faf2ed"/><stop offset=".6" stop-color="#e7d4d8"/>
          <stop offset="1" stop-color="#9eb4d0"/>
        </radialGradient>
        <linearGradient id="nha_brain" x1="0" y1="0" x2="0" y2="1">
          <stop stop-color="#ffd9d3"/><stop offset=".5" stop-color="#f4a098"/>
          <stop offset="1" stop-color="#de7579"/>
        </linearGradient>
        <radialGradient id="nha_lung" cx="37%" cy="35%" r="75%">
          <stop stop-color="#ffddc7"/><stop offset=".55" stop-color="#f3aa95"/>
          <stop offset="1" stop-color="#d67a75"/>
        </radialGradient>
        <linearGradient id="nha_heart" x1="0" y1="0" x2=".8" y2="1">
          <stop stop-color="#ffb09a"/><stop offset=".35" stop-color="#f36956"/>
          <stop offset="1" stop-color="#c92b34"/>
        </linearGradient>
        <radialGradient id="nha_liver" cx="32%" cy="26%" r="80%">
          <stop stop-color="#ffc27e"/><stop offset=".48" stop-color="#ed8651"/>
          <stop offset="1" stop-color="#ad4938"/>
        </radialGradient>
        <linearGradient id="nha_stomach" x1="0" y1="0" x2="1" y2="1">
          <stop stop-color="#ffe4b7"/><stop offset=".5" stop-color="#f7ba83"/>
          <stop offset="1" stop-color="#dc8567"/>
        </linearGradient>
        <linearGradient id="nha_kidney" x1="0" y1="0" x2="1" y2="1">
          <stop stop-color="#f3b9ae"/><stop offset=".5" stop-color="#d8827c"/>
          <stop offset="1" stop-color="#a64f64"/>
        </linearGradient>
        <linearGradient id="nha_intestine" x1="0" y1="0" x2="1" y2="0">
          <stop stop-color="#f9d9a0"/><stop offset=".5" stop-color="#ffc790"/>
          <stop offset="1" stop-color="#df8b63"/>
        </linearGradient>
      </defs>

      <!-- Continuous translucent outline, including fingers and upper thighs. -->
      <path d="M73 40 Q73 49 67 55 C62 59 53 59 48 63
        C38 68 35 79 33 92 L29 113 Q25 125 25 138
        C24 149 20 159 19 168 L14 176 9 181 3 191
        Q1 195 4 195 L10 188 6 201 Q5 205 8 205 L13 192
        L10 208 Q11 212 14 208 L18 193 16 209 Q17 212 20 207
        L23 193 23 204 Q25 207 27 201 L29 189 Q30 182 26 176
        C29 165 36 154 39 144 Q43 135 43 126 L48 108
        Q51 103 52 96 C51 111 48 122 49 136 L51 152
        C49 160 46 170 46 181 L43 204 43 222 68 222
        L72 202 Q75 191 81 189 Q87 191 90 202 L94 222 119 222
        L119 204 116 181 C116 170 113 160 111 152 L113 136
        C114 122 111 111 110 96 Q111 103 114 108 L119 126
        Q119 135 123 144 C126 154 133 165 136 176
        Q132 182 133 189 L135 201 Q137 207 139 204 L139 193
        L142 207 Q145 212 146 209 L144 193 148 208
        Q151 212 152 208 L149 192 154 205 Q157 205 156 201
        L152 188 158 195 Q161 195 159 191 L153 181 148 176 143 168
        C142 159 138 149 137 138 Q137 125 133 113 L129 92
        C127 79 124 68 114 63 C109 59 100 59 95 55 Q89 49 89 40 Z"
        fill="url(#nha_skin)" stroke="#3c567d" stroke-width="1.2"
        stroke-linejoin="round"/>
      <path d="M71 48 Q69 57 55 62 C44 67 40 79 39 92
        Q36 111 32 125 C31 143 26 162 22 175
        M91 48 Q93 57 107 62 C118 67 122 79 123 92
        Q126 111 130 125 C131 143 136 162 140 175"
        fill="none" stroke="#faf8f8" stroke-width="3.1" opacity=".86"/>
      <path d="M48 79 Q39 91 40 107 M114 79 Q123 91 122 107
        M31 131 Q29 145 26 155 M131 131 Q133 145 136 155
        M51 154 Q53 168 46 183 M111 154 Q109 168 116 183
        M57 183 Q62 196 55 222 M105 183 Q100 196 107 222"
        fill="none" stroke="#8b9db9" stroke-width="1.15" opacity=".7"/>
      <path d="M54 66 Q65 60 74 63 M88 63 Q97 60 108 66
        M49 180 Q63 174 75 187 M113 180 Q99 174 87 187
        M53 190 Q56 205 51 220 M109 190 Q106 205 111 220"
        fill="none" stroke="#fff9ef" stroke-width="3" opacity=".8"/>
      <path d="M53 94 C48 113 55 120 51 138 M109 94 C114 113 107 120 111 138
        M51 143 Q56 154 59 161 M111 143 Q106 154 103 161"
        fill="none" stroke="#b9adbf" stroke-width="1.1" opacity=".7"/>

      <!-- Head and neck: cranial dome, ears, face and carotid branches. -->
      <path d="M66 23 Q62 19 63 29 Q63 35 68 35 M96 23 Q100 19 99 29 Q99 35 94 35"
        fill="#ccd5e5" stroke="#526383" stroke-width="1"/>
      <path d="M66 14 C66 3 74 1 81 2 C88 1 96 5 96 14 L94 34
        C92 42 87 48 81 49 C75 48 70 42 68 34 Z"
        fill="url(#nha_face)" stroke="#4b6083" stroke-width="1.25"/>
      <path d="M69 14 C70 6 76 5 81 5 C87 5 92 8 93 14 L92 25
        Q86 29 81 26 Q76 29 70 25 Z" fill="url(#nha_brain)" stroke="#be7b8d" stroke-width=".65"/>
      <path d="M81 6 81 26 M77 8 Q72 10 74 14 Q70 16 74 20 L73 24
        M85 8 Q90 10 88 14 Q92 16 88 20 L89 24
        M77 10 78 23 M85 10 84 23"
        fill="none" stroke="#ffe6db" stroke-width=".9" stroke-linecap="round"/>
      <path d="M70 28 Q74 26 77 29 M85 29 Q88 26 92 28 M72 31 76 31 M86 31 90 31
        M81 28 79 36 83 36 M77 41 Q81 43 85 41"
        fill="none" stroke="#66708b" stroke-width=".85" stroke-linecap="round"/>
      <path d="M72 35 Q73 40 77 43 M90 35 Q89 40 85 43"
        fill="none" stroke="#f9e6de" stroke-width="1.4"/>
      <path d="M77 47 76 62 M85 47 86 62" fill="none" stroke="#d75b61" stroke-width="1.6"/>
      <path d="M74 45 73 60 M88 45 89 60" fill="none" stroke="#548ac8" stroke-width="1.4"/>
      <path d="M81 46 81 77" fill="none" stroke="#9bb5ca" stroke-width="4.8"/>
      <path d="M79 49 83 49 M79 53 83 53 M79 57 83 57 M79 61 83 61
        M79 65 83 65 M79 69 83 69"
        fill="none" stroke="#f6e7df" stroke-width="1"/>

      <!-- Rib cage and paired lung lobes. -->
      <path d="M52 72 Q62 65 74 69 M88 69 Q100 65 110 72
        M51 79 Q63 71 74 75 M88 75 Q99 71 111 79
        M50 86 Q61 79 74 82 M88 82 Q101 79 112 86
        M49 94 Q61 87 73 90 M89 90 Q101 87 113 94
        M49 102 Q60 97 72 99 M90 99 Q102 97 113 102
        M50 110 Q60 108 71 107 M91 107 Q102 108 112 110"
        fill="none" stroke="#f7eff0" stroke-width="1.7" opacity=".95"/>
      <path d="M63 70 C54 71 52 79 50 89 Q47 100 50 112
        Q60 119 72 111 L75 91 74 77 Q72 68 63 70 Z"
        fill="url(#nha_lung)" stroke="#ae6c74" stroke-width="1.05"/>
      <path d="M99 70 C108 71 110 79 112 89 Q115 100 112 112
        Q102 119 91 111 L88 100 Q95 95 88 88 L88 77 Q90 68 99 70 Z"
        fill="url(#nha_lung)" stroke="#ae6c74" stroke-width="1.05"/>
      <path d="M80 76 67 85 62 97 M81 77 95 85 101 98
        M67 85 57 89 M66 89 72 98 M95 85 105 89 M96 89 91 98"
        fill="none" stroke="#edd7ca" stroke-width="2.4" stroke-linecap="round"/>
      <path d="M54 98 Q63 95 72 98 M91 99 Q102 95 109 100
        M56 107 Q62 105 69 107 M95 108 Q102 105 109 108"
        fill="none" stroke="#e6a69a" stroke-width=".8"/>

      <!-- Heart and great vessels, shown on the front of the lung pair. -->
      <path d="M77 87 C77 82 84 81 87 87 C92 84 96 88 97 95
        C98 103 89 111 82 116 C75 111 72 102 73 95 Q72 90 77 87 Z"
        fill="url(#nha_heart)" stroke="#9e343d" stroke-width="1.1"/>
      <path d="M84 89 C80 95 81 104 85 111 M84 97 91 92 M82 102 76 99"
        fill="none" stroke="#f6b096" stroke-width=".95"/>
      <path d="M87 90 V79 Q87 73 93 73 Q99 73 99 80"
        fill="none" stroke="#b6373b" stroke-width="4.6"/>
      <path d="M87 90 V79 Q87 75 92 75" fill="none" stroke="#ff9c82" stroke-width="1.5"/>
      <path d="M80 90 77 82 70 80 M84 87 84 77"
        fill="none" stroke="#497fba" stroke-width="3.1" stroke-linecap="round"/>

      <!-- Liver, stomach, pancreatic tissue and renal pair. -->
      <path d="M53 117 C59 110 70 111 80 114 Q90 113 97 118
        L91 127 Q84 131 77 130 Q72 136 62 135 L53 130 Q49 125 53 117 Z"
        fill="url(#nha_liver)" stroke="#914631" stroke-width="1.2"/>
      <path d="M76 115 Q77 123 72 131 M55 120 Q62 115 70 118"
        fill="none" stroke="#ffca8f" stroke-width="1.1" stroke-linecap="round"/>
      <path d="M76 129 Q71 133 73 139 Q78 141 80 135 Q81 131 76 129 Z"
        fill="#7da04a" stroke="#526d31" stroke-width=".7"/>
      <path d="M99 113 Q96 119 97 125 C98 130 101 134 96 139
        C92 143 86 139 83 142 C78 150 89 154 98 149
        C109 143 111 135 107 124 L105 116 Z"
        fill="url(#nha_stomach)" stroke="#ad674d" stroke-width="1.1"/>
      <path d="M101 118 Q99 125 103 131 Q108 140 98 145 Q91 149 85 146"
        fill="none" stroke="#ffe6b9" stroke-width="1.6" stroke-linecap="round"/>
      <path d="M108 123 Q115 126 112 139 L108 143 106 135 Z"
        fill="#aa7194" stroke="#765272" stroke-width=".8"/>
      <path d="M69 139 Q78 135 88 139 L96 142 92 146 78 143 68 144 Z"
        fill="#e9c382" stroke="#b28b51" stroke-width=".75"/>
      <path d="M55 137 C48 136 47 146 51 153 Q56 159 61 153
        Q63 148 59 146 Q55 145 57 142 Q59 139 55 137 Z"
        fill="url(#nha_kidney)" stroke="#885364" stroke-width=".8"/>
      <path d="M107 137 C114 136 115 146 111 153 Q106 159 101 153
        Q99 148 103 146 Q107 145 105 142 Q103 139 107 137 Z"
        fill="url(#nha_kidney)" stroke="#885364" stroke-width=".8"/>
      <path d="M56 147 Q64 158 73 183 M106 147 Q98 158 89 183"
        fill="none" stroke="#ceb578" stroke-width="1.2"/>

      <!-- Colon encloses independently curved small-intestine loops. -->
      <path d="M64 147 C60 147 58 153 59 160 L59 173
        Q58 180 64 182 L97 182 Q103 181 103 174 L103 157 Q104 150 98 148"
        fill="none" stroke="#a96149" stroke-width="7.5" stroke-linecap="round"/>
      <path d="M64 147 C60 147 60 155 61 160 L61 173
        Q60 179 65 180 L96 180 Q101 179 101 173 L101 157 Q102 151 98 150"
        fill="none" stroke="url(#nha_intestine)" stroke-width="5" stroke-linecap="round"/>
      <path d="M59 157 63 158 M59 165 63 166 M60 174 64 175 M69 178 69 182
        M78 178 78 182 M87 178 87 182 M96 177 97 181 M100 170 104 169 M100 160 104 159"
        fill="none" stroke="#d59163" stroke-width=".85"/>
      <path d="M68 150 C74 145 81 154 87 150 C94 146 98 151 95 155
        C91 159 79 151 71 156 C64 160 69 164 77 160
        C86 155 99 161 94 164 C88 169 79 161 71 165
        C63 169 71 174 79 169 C87 165 98 169 95 173
        C92 177 82 172 75 176"
        fill="none" stroke="#b57552" stroke-width="4.6" stroke-linecap="round"/>
      <path d="M68 150 C74 145 81 154 87 150 C94 146 98 151 95 155
        C91 159 79 151 71 156 C64 160 69 164 77 160
        C86 155 99 161 94 164 C88 169 79 161 71 165
        C63 169 71 174 79 169 C87 165 98 169 95 173
        C92 177 82 172 75 176"
        fill="none" stroke="#ffd19c" stroke-width="2.7" stroke-linecap="round"/>
      <path d="M75 184 Q81 180 87 184 L87 190 Q81 197 75 190 Z"
        fill="#edb7b4" stroke="#b07b86" stroke-width=".8"/>

      <!-- Paired red/blue vessels and fine distal branching. -->
      <g fill="none" stroke-linecap="round" stroke-linejoin="round">
        <path d="M86 85 V67 Q85 59 77 60 L57 66 Q48 69 45 82
          L37 110 Q35 125 30 137 Q28 158 23 175
          M87 62 107 66 Q116 69 119 82 L127 110 Q129 125 134 137 Q136 158 141 175
          M86 113 85 135 84 160 83 183 Q76 190 67 196 L60 222
          M83 183 Q90 190 99 196 L106 222"
          stroke="#fff2eb" stroke-width="4.1" opacity=".9"/>
        <path d="M86 85 V67 Q85 59 77 60 L57 66 Q48 69 45 82
          L37 110 Q35 125 30 137 Q28 158 23 175
          M87 62 107 66 Q116 69 119 82 L127 110 Q129 125 134 137 Q136 158 141 175
          M86 113 85 135 84 160 83 183 Q76 190 67 196 L60 222
          M83 183 Q90 190 99 196 L106 222"
          stroke="#e64a43" stroke-width="1.85"/>
        <path d="M77 86 V67 L56 70 Q47 75 44 88 L38 113
          Q35 129 33 141 L26 176 M79 65 107 70 Q116 75 120 88
          L126 113 Q129 129 131 141 L138 176
          M78 115 78 137 78 160 79 183 63 198 55 222
          M79 183 95 198 101 222"
          stroke="#f4f8fc" stroke-width="3.3" opacity=".9"/>
        <path d="M77 86 V67 L56 70 Q47 75 44 88 L38 113
          Q35 129 33 141 L26 176 M79 65 107 70 Q116 75 120 88
          L126 113 Q129 129 131 141 L138 176
          M78 115 78 137 78 160 79 183 63 198 55 222
          M79 183 95 198 101 222"
          stroke="#397fd0" stroke-width="1.55"/>
        <path d="M45 86 51 90 M119 86 113 90 M38 111 34 119 34 132
          M126 111 130 119 130 132 M23 175 15 184 12 197
          M23 176 21 190 19 203 M141 175 149 184 152 197
          M141 176 143 190 145 203 M67 196 70 212 68 222
          M99 196 96 212 98 222 M85 132 98 139 M84 145 62 146"
          stroke="#ed6a5f" stroke-width=".95"/>
        <path d="M26 176 23 185 23 196 M26 178 17 187 15 202
          M138 176 141 185 141 196 M138 178 147 187 149 202
          M63 198 66 206 M95 198 92 206 M78 138 63 143"
          stroke="#397dbb" stroke-width=".9"/>
      </g>
    </g>
    ''')


def _organs() -> ET.Element:
    return ET.fromstring(r'''
    <g xmlns="http://www.w3.org/2000/svg" id="native_organs">
      <title>Hand-drawn liver and renal organ illustrations</title>
      <defs>
        <radialGradient id="nho_liver" cx="31%" cy="22%" r="87%">
          <stop stop-color="#ffc29a"/><stop offset=".45" stop-color="#eb865c"/>
          <stop offset="1" stop-color="#b74735"/>
        </radialGradient>
        <linearGradient id="nho_bile" x1="0" y1="0" x2="1" y2="1">
          <stop stop-color="#b2c06c"/><stop offset="1" stop-color="#668744"/>
        </linearGradient>
        <radialGradient id="nho_renal" cx="35%" cy="25%" r="80%">
          <stop stop-color="#ffd4d2"/><stop offset=".5" stop-color="#d98a9d"/>
          <stop offset="1" stop-color="#a45479"/>
        </radialGradient>
        <linearGradient id="nho_medulla" x1="0" y1="0" x2="1" y2="1">
          <stop stop-color="#ffead8"/><stop offset="1" stop-color="#e8baa7"/>
        </linearGradient>
      </defs>
      <g stroke-linecap="round" stroke-linejoin="round">
        <path d="M699 119 C704 107 717 102 730 106
          C741 106 750 110 763 115 Q765 116 763 120
          L754 132 Q748 137 742 139 L738 149 Q735 154 731 152
          L723 149 717 142 Q708 146 702 141 C695 137 694 128 699 119 Z"
          fill="url(#nho_liver)" stroke="#543b3c" stroke-width="1.6"/>
        <path d="M724 107 Q721 116 723 122 Q726 127 722 143
          M724 128 Q733 123 742 127 Q744 132 741 138"
          fill="none" stroke="#a1543d" stroke-width="1.15"/>
        <path d="M703 119 Q710 110 719 110 M731 109 Q745 110 755 115"
          fill="none" stroke="#ffd1ad" stroke-width="2.1" opacity=".8"/>
        <path d="M736 136 C740 139 738 146 735 150 Q731 151 730 146
          Q728 140 733 137 Z" fill="url(#nho_bile)" stroke="#516139" stroke-width="1"/>
        <path d="M735 138 738 132" fill="none" stroke="#6c8246" stroke-width="1.2"/>

        <path d="M711 174 C702 175 696 187 696 199 C696 214 703 226 712 230
          Q720 234 722 227 Q724 221 718 216 C710 212 709 204 715 199
          C719 196 722 191 722 185 Q722 173 715 173 Z"
          fill="url(#nho_renal)" stroke="#3e3b5d" stroke-width="1.5"/>
        <path d="M748 174 C757 175 763 187 763 199 C763 214 756 226 747 230
          Q739 234 737 227 Q735 221 741 216 C749 212 750 204 744 199
          C740 196 737 191 737 185 Q737 173 744 173 Z"
          fill="url(#nho_renal)" stroke="#3e3b5d" stroke-width="1.5"/>
        <path d="M709 183 C703 187 701 196 703 204 Q704 214 711 219
          L715 215 Q708 206 714 198 Q719 190 716 185 Z
          M750 183 C756 187 758 196 756 204 Q755 214 748 219
          L744 215 Q751 206 745 198 Q740 190 743 185 Z"
          fill="url(#nho_medulla)" stroke="#b8869b" stroke-width=".7"/>
        <path d="M705 190 711 194 M704 200 710 202 M707 212 712 208
          M754 190 748 194 M755 200 749 202 M752 212 747 208"
          fill="none" stroke="#dba9aa" stroke-width=".9"/>
        <path d="M718 204 Q725 212 725 227 L724 239
          M741 204 Q734 212 734 227 L735 239"
          fill="none" stroke="#a88b55" stroke-width="4.1"/>
        <path d="M718 204 Q725 212 725 227 L724 239
          M741 204 Q734 212 734 227 L735 239"
          fill="none" stroke="#f5daa0" stroke-width="2.8"/>
        <path d="M729 196 729 228 M729 202 716 201 M729 202 743 201"
          fill="none" stroke="#f7ede6" stroke-width="4.3"/>
        <path d="M729 196 729 228 M729 202 716 201 M729 202 743 201"
          fill="none" stroke="#d85661" stroke-width="2.2"/>
        <path d="M732 196 732 226 M732 206 718 206 M732 206 741 206"
          fill="none" stroke="#5185ba" stroke-width="1.8"/>
        <path d="M703 184 Q706 177 713 177 M756 184 Q753 177 746 177"
          fill="none" stroke="#ffe2d8" stroke-width="1.8" opacity=".9"/>
      </g>
    </g>
    ''')


def native_anatomy(name: str) -> ET.Element:
    """Return 'human' or 'organs' as editable SVG in global canvas coordinates."""
    if name == "human":
        return _human()
    if name == "organs":
        return _organs()
    raise ValueError(f"Unknown anatomy component: {name!r}")
