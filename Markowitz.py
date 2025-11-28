    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """

        # 只對 sector（排除 SPY）做配置
        sector_cols = assets

        # 動能與波動視窗（可以之後自行微調）
        mom_window = max(self.lookback, 60)   # 用至少 60 天做動能
        vol_window = max(self.lookback // 2, 20)  # 用較短視窗估波動
        top_k = 3                             # 每次選動能最高的 3 檔

        # 使用 log-return 當作動能基礎（和 Ref1 的 product 寫法不同）
        sector_ret = self.returns[sector_cols]
        log_ret = np.log1p(sector_ret)

        # rolling 動能分數：sum of log(1+r)
        rolling_mom = (
            log_ret.rolling(window=mom_window, min_periods=mom_window)
                   .sum()
        )

        # rolling 波動度：用原始日報酬的標準差
        rolling_vol = (
            sector_ret.rolling(window=vol_window, min_periods=vol_window)
                      .std()
        )

        for idx, date in enumerate(self.price.index):
            # 一開始資料不足就跳過，之後由 ffill 補齊
            if idx < max(mom_window, vol_window):
                continue

            mom_today = rolling_mom.loc[date]
            vol_today = rolling_vol.loc[date]

            # 若今天沒有有效數值，跳過
            if mom_today.isna().all() or vol_today.isna().all():
                continue

            # 只保留動能為正的 sector，若沒有就用全部
            positive = mom_today[mom_today > 0]
            if positive.empty:
                ranked = mom_today.sort_values(ascending=False)
            else:
                ranked = positive.sort_values(ascending=False)

            # 取動能最高的 top_k 檔
            selected = ranked.head(top_k).index

            # 取出這幾檔的短期波動度
            vol_sel = vol_today.loc[selected].replace(0.0, np.nan)

            # inverse-vol 權重分數
            inv_vol = 1.0 / vol_sel
            inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            score_sum = inv_vol.sum()
            if score_sum > 0:
                weights_sel = inv_vol / score_sum
            else:
                # fallback：如果波動度有問題，就改成等權
                weights_sel = pd.Series(1.0 / len(selected), index=selected)

            # 建立當日完整權重向量：只在 selected 上有部位
            today_w = pd.Series(0.0, index=self.price.columns)
            today_w.loc[selected] = weights_sel

            # SPY 權重強制為 0
            today_w.loc[self.exclude] = 0.0

            # 寫入當日權重
            self.portfolio_weights.loc[date, :] = today_w

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)
