# ==========================================
# 1. SETUP AND LOAD LIBRARIES
# ==========================================
library(cluster)  
library(plot3D)   
library(GGally)   
library(ggplot2)  

print("Loading and scaling data...")
df <- read.csv("2026_soc_ready_for_clustering.csv")
df_scaled <- df

# Scale the structural axes and apply the EMD/Intersection weights
df_scaled[, c("burstiness", "perplexity", "ttr")] <- scale(df[, c("burstiness", "perplexity", "ttr")])
df_scaled$burstiness <- df_scaled$burstiness * 1.00
df_scaled$perplexity <- df_scaled$perplexity * 1.14
df_scaled$ttr <- df_scaled$ttr * 1.65

# ==========================================
# 2. RUN K-MEDOIDS ALGORITHM
# ==========================================
print("Running PAM (K-Medoids) algorithm...")
pam_result <- pam(
  df_scaled[, c("burstiness", "perplexity", "ttr")], 
  k = 2, 
  metric = "manhattan"
)

# Attach results
df_scaled$predicted_cluster <- as.factor(pam_result$clustering)

# CRITICAL FIX: Force exact text labels so ggplot maps colors perfectly
df_scaled$true_label <- factor(df_scaled$label, levels = c("human", "ai"))
medoids <- pam_result$medoids

# Print the confusion matrix to the console
print("Confusion Matrix: True Labels vs K-Medoids Clusters")
table(True_Origin = df_scaled$true_label, Predicted_Cluster = df_scaled$predicted_cluster)

# ==========================================
# 3. PLOT 1: TRUE ORIGIN PAIRPLOT
# ==========================================
print("Generating True Origin Pairplot...")
pairplot_true <- ggpairs(
  df_scaled, 
  columns = c("burstiness", "perplexity", "ttr"),
  mapping = aes(color = true_label, alpha = 0.4), 
  columnLabels = c("Burstiness (Weighted)", "Perplexity (Weighted)", "TTR (Weighted)"),
  upper = list(continuous = wrap("cor", size = 4)),
  lower = list(continuous = wrap("points", size = 1.5, stroke = 0))
) + 
  scale_color_manual(values = c("human" = "#636efa", "ai" = "#ef553b")) + 
  scale_fill_manual(values = c("human" = "#636efa", "ai" = "#ef553b")) +
  theme_minimal() +
  labs(title = "Reality: Human vs. AI True Origins")

print(pairplot_true)

# ==========================================
# 4. PLOT 2: PREDICTED CLUSTERS PAIRPLOT
# ==========================================
print("Generating Predicted Clusters Pairplot...")
pairplot_pred <- ggpairs(
  df_scaled, 
  columns = c("burstiness", "perplexity", "ttr"),
  mapping = aes(color = predicted_cluster, alpha = 0.4), 
  columnLabels = c("Burstiness", "Perplexity", "TTR"),
  upper = list(continuous = wrap("cor", size = 4)),
  lower = list(continuous = wrap("points", size = 1.5, stroke = 0))
) + 
  scale_color_manual(values = c("1" = "#00b4d8", "2" = "#7209b7")) + 
  scale_fill_manual(values = c("1" = "#00b4d8", "2" = "#7209b7")) +
  theme_minimal() +
  labs(title = "Algorithm's View: K-Medoids Predicted Clusters")

# R will draw this on top of the first one. Use the back arrow in the Plots pane to see both!
print(pairplot_pred)

# ==========================================
# 5. PLOT 3 & 4: SIDE-BY-SIDE 3D GEOMETRY
# ==========================================
print("Opening new window for side-by-side 3D rendering...")

# Setup specific colors for both views with transparency (alpha)
colors_true <- ifelse(df_scaled$true_label == "human", 
                      rgb(0.39, 0.44, 0.98, alpha = 0.3),  # Blue
                      rgb(0.93, 0.33, 0.23, alpha = 0.3))  # Red

colors_pred <- ifelse(df_scaled$predicted_cluster == "1", 
                      rgb(0.0, 0.7, 0.85, alpha = 0.3),    # Teal
                      rgb(0.45, 0.04, 0.72, alpha = 0.3))  # Purple

# Open a single wide window
dev.new(width = 14, height = 7) 

# Split the canvas into 1 row and 2 columns
par(mfrow = c(1, 2))

# --- LEFT SIDE: True Origin ---
scatter3D(
  x = df_scaled$burstiness, y = df_scaled$perplexity, z = df_scaled$ttr,
  colvar = NULL, col = colors_true, 
  pch = 16, cex = 0.8, cex.lab = 0.9, cex.axis = 0.8,
  phi = 20, theta = 45, 
  xlab = "Burstiness", ylab = "Perplexity", zlab = "TTR",
  main = "Reality: True Origin",
  ticktype = "detailed", bty = "b2"
)
scatter3D(x = medoids[, "burstiness"], y = medoids[, "perplexity"], z = medoids[, "ttr"],
          colvar = NULL, col = c("black", "black"), pch = 4, cex = 2.5, lwd = 3, add = TRUE)

# --- RIGHT SIDE: Predicted Clusters ---
scatter3D(
  x = df_scaled$burstiness, y = df_scaled$perplexity, z = df_scaled$ttr,
  colvar = NULL, col = colors_pred, 
  pch = 16, cex = 0.8, cex.lab = 0.9, cex.axis = 0.8,
  phi = 20, theta = 45, 
  xlab = "Burstiness", ylab = "Perplexity", zlab = "TTR",
  main = "Algorithm: Predicted Clusters",
  ticktype = "detailed", bty = "b2"
)
scatter3D(x = medoids[, "burstiness"], y = medoids[, "perplexity"], z = medoids[, "ttr"],
          colvar = NULL, col = c("black", "black"), pch = 4, cex = 2.5, lwd = 3, add = TRUE)

# Reset the canvas layout back to default just to be safe
par(mfrow = c(1, 1))

print("Pipeline complete! All plots generated successfully.")