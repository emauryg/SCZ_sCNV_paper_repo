# Investigating ABCB11 in neurons
library(qs)
library(rliger)
library(RColorBrewer)
library(gridExtra)

setwd('~/ABCB11/')
#allen.mat <-  read.csv('matrix.csv')
#rownames(allen.mat) <- allen.mat[,1]
#allen.mat$sample_name <- NULL

#allen.mat.sparse <- as(as.matrix(allen.mat),Class = 'dgCMatrix')

allen.mat.sparse <- qread('sparse_allenmatrix.qs')

metadata.allen <- read.csv("metadata.csv")

out.matrix.divided <- lapply(unique(metadata.allen$external_donor_name_label),function(x){
  idx.use <- metadata.allen[metadata.allen$external_donor_name_label == x,]$sample_name
  return(t(allen.mat.sparse[idx.use,]))
})
names(out.matrix.divided) <- unique(metadata.allen$external_donor_name_label)

allen.liger <- rliger::createLiger(raw.data = out.matrix.divided)
allen.liger@clusters <- as.factor(metadata.allen[match(rownames(allen.liger@cell.data),
                                             metadata.allen$sample_name),]$cell_type_alias_label)
allen.liger <- rliger::normalize(allen.liger)

tsne.use <- read.csv('tsne.csv')
tsne.use <- tsne.use[match(rownames(allen.liger@cell.data),tsne.use$sample_name),]
rownames(tsne.use) <- tsne.use$sample_name
tsne.use$sample_name <- NULL
allen.liger@tsne.coords <- as.matrix(tsne.use)
qsave(allen.liger,"~/ABCB11/allen_liger.qs")

tsne.plot <- as.data.frame(allen.liger@tsne.coords)
tsne.plot$clusters <- allen.liger@clusters
mycolors <- colorRampPalette(brewer.pal(8, "Dark2"))(length(unique(allen.liger@clusters)))


pdf('~/ABCB11/allen_umap.pdf',useDingbats = F)
ggplot(tsne.plot,aes(x = tsne_1,y = tsne_2,color = clusters)) + geom_point(size = 0.3) + 
    theme_cowplot() + theme(legend.position = 'none',axis.text = element_blank(),
                            axis.title = element_blank(),axis.ticks = element_blank()
                            ,axis.line = element_blank())
dev.off()


tsne.plot$gene <- gene.abcb11
p1 <- rliger::plotGene(allen.liger,return.plots = T,plot.by = 'none',gene = 'ABCB11',pt.size = 0.7)
pdf('~/ABCB11/allen_abcb11.pdf',useDingbats = F)
p1 +theme_cowplot() + theme(legend.position = 'none',axis.text = element_blank(),
                          axis.title = element_blank(),axis.ticks = element_blank()
                          ,axis.line = element_blank())
dev.off()

all.obj <- qread('~/DA/allobj_liger.qs')

out.plots <- lapply(names(all.obj),function(x){
  liger.use <- all.obj[[x]]
  pd.meta.use <- read.csv('~/Finnish_analysis/PD_snrna_metadata.csv')
  liger.use@cell.data$status <- pd.meta.use[match(liger.use@cell.data$dataset,
                                                 pd.meta.use$Donor.ID),]$Status
  liger.use <- subsetLiger(liger.use,cells.use = rownames(liger.use@cell.data[which(
    liger.use@cell.data$status %in% c('C','CA')),]))
  p1 <- rliger::plotGene(liger.use,gene = 'ABCB11',plot.by = 'none',return.plots = T)
  p1 <- p1 + theme(axis.text = element_blank(),
                   axis.title = element_blank(),axis.ticks = element_blank()
                   ,axis.line = element_blank())
  return(p1)
})

pdf('~/ABCB11/plots_SN.pdf',useDingbats = F)
do.call('grid.arrange',out.plots)
dev.off()


striatum.liger <- qread('striatum_ligeruse.qs')
striatum.liger.use <- rliger::createLiger(raw.data = striatum.liger@raw.data)
striatum.liger.use@tsne.coords <- striatum.liger@tsne.coords
striatum.liger.use@clusters <- striatum.liger@clusters

p1 <- rliger::plotGene(striatum.liger.use,gene = 'ABCB11',plot.by = 'none',return.plots = T)
pdf('~/ABCB11/plots_striatum.pdf',useDingbats = F)
p1 + theme(axis.text = element_blank(),
             axis.title = element_blank(),axis.ticks = element_blank()
             ,axis.line = element_blank())
dev.off()


p.out <- lapply(all.obj,function(x){
  liger.use <- x
  pd.meta.use <- read.csv('~/Finnish_analysis/PD_snrna_metadata.csv')
  liger.use@cell.data$status <- pd.meta.use[match(liger.use@cell.data$dataset,
                                                  pd.meta.use$Donor.ID),]$Status
  liger.use <- subsetLiger(liger.use,cells.use = rownames(liger.use@cell.data[which(
    liger.use@cell.data$status %in% c('C','CA')),]))
  
df.hpc <- data.frame('gene' = getGeneValues(liger.use@norm.data,gene = 'ABCB11'),
                     'dataset' = liger.use@cell.data$dataset,'cluster' = liger.use@clusters)
aggregate(gene ~ cluster + dataset,df.hpc,mean)
})
names(p.out) <- names(all.obj)
p.sn <- bind_rows(p.out)


df.allen <- data.frame('gene' = getGeneValues(allen.liger@norm.data,gene = 'ABCB11'),
                     'dataset' = allen.liger@cell.data$dataset,'cluster' = allen.liger@clusters)
p.allen <- aggregate(gene ~ cluster + dataset,df.allen,mean)

df.striatum <- data.frame('gene' = getGeneValues(striatum.liger.use@norm.data,gene = 'ABCB11'),
                       'dataset' = striatum.liger.use@cell.data$dataset,'cluster' = striatum.liger.use@clusters)
p.striatum <- aggregate(gene ~ cluster + dataset,df.striatum,mean)

#######################
## Plot for panel A ###
#######################

p.plot <- bind_rows(list('Dorsal_striatum' = p.striatum,
                         'SN'= p.sn,'M1' = p.allen),.id = 'region')

pdf('~/ABCB11/normexpr.pdf',useDingbats = F,width = 30,height = 20)
ggplot(p.plot2[order(p.plot2$gene,decreasing = T),],aes(x = cluster,y = gene)) + theme_cowplot()+
  geom_point() + geom_boxplot() + ylim(NA,0.0001) +
  theme(axis.text.x = element_text(angle = 45,hjust = 1)) + facet_wrap(~region,scales = 'free_x',nrow = 3) +
  ylab('Normalized gene expression') + xlab('Cell type') 
dev.off()

t1 <- aggregate(gene ~ cluster,p.plot2,mean)
clusters.use <- t1[which(t1$gene > 1e-6),]$cluster

p.plot2 <- p.plot[p.plot$region %in% c('M1','Dorsal_striatum','SN'),]
p.plot2 <- p.plot2[which( (p.plot2$cluster %in% clusters.use & p.plot2$region %in% c('M1','SN') ) | 
                           (p.plot2$region == 'Dorsal_striatum')),]
p.plot2$cluster <- fct_reorder(p.plot2$cluster,p.plot2$gene,.fun = median)
p.plot2$region <- as.factor(p.plot2$region)
p.plot2$region <- factor(p.plot2$region,levels = c('SN','M1','Dorsal_striatum'))
p.plot2$lognormgene <- log10(100000*p.plot2$gene + 1)
levels(p.plot2$region) <- c('Substantia nigra','M1 motor cortex','Dorsal striatum')
pdf('~/ABCB11/normexpr_pruned.pdf',useDingbats = F,height = 15,width = 18)
ggplot(p.plot2,aes(x = cluster,y = lognormgene,fill = cluster)) + theme_cowplot()+
  geom_jitter(width = 0.2) + geom_boxplot(alpha = 0.3,outlier.shape = NA) +
  theme(axis.text.x = element_text(angle = 45,hjust = 1,size = 20),legend.position = 'none',
        axis.text.y = element_text(size = 20),strip.text = element_text(size=30),
        strip.background = element_blank()) + 
  facet_wrap(~region,scales = 'free_y',ncol = 3) +
  ylab('Normalized gene expression') + xlab('Cell type') + coord_flip()
dev.off()

###############################
### Plot UMAP for panel B #####
###############################

library(MetBrewer)
liger.use <- all.obj$da
pd.meta.use <- read.csv('~/Finnish_analysis/PD_snrna_metadata.csv')
liger.use@cell.data$status <- pd.meta.use[match(liger.use@cell.data$dataset,
                                                pd.meta.use$Donor.ID),]$Status
liger.use <- subsetLiger(liger.use,cells.use = rownames(liger.use@cell.data[which(
  liger.use@cell.data$status %in% c('C','CA')),]))

tsne_df <- data.frame(liger.use@tsne.coords)
colnames(tsne_df) <- c("Dim1", "Dim2")
c_names <- names(liger.use@clusters)
clusters <- liger.use@clusters
tsne_df[['Cluster']] <- clusters[c_names]
centers <- tsne_df %>% group_by(.data[['Cluster']]) %>% summarize(
  tsne1 = median(x = .data[['Dim1']]),
  tsne2 = median(x = .data[['Dim2']])
)

p1 <- plotByDatasetAndCluster(liger.use,return.plots = T,pt.size = 0.8,text.size = 0)
colors.use <- met.brewer(name = 'Egypt', n = 10)
p.umap.daneurons <- p1[[2]] + scale_color_manual(values = colors.use)
pdf('~/ABCB11/SN_umap_panelB.pdf',useDingbats = F)
p.umap.daneurons + theme(legend.position = 'none',
                         axis.text = element_blank(),
                         axis.ticks = element_blank(),
                         axis.title = element_blank(),
                         axis.line = element_blank()) +
  geom_label_repel(data = centers, mapping = aes_string(label = 'Cluster'), 
                   colour = "black", size = 3.5)
dev.off()

seurat.da <- ligerToSeurat(liger.use)
VlnPlot(seurat.da,features = c('CALB1'),cols = colors.use)
VlnPlot(seurat.da,features = c('ABCB11'),cols = colors.use) + theme(legend.position = 'none') + 
  coord_flip()
p1 <- DotPlot(seurat.da,features = c('CALB1','ABCB11'),cols = c('lightgrey',colors.use[4]))
p1$data$id <- fct_reorder(p1$data$id,p1$data$pct.exp)
pdf('~/ABCB11/SN_dotplot_panelB.pdf',useDingbats = F,width = 3,height = 8 )
p1 + theme(legend.position = 'bottom',axis.text.x = element_text(angle = 45,hjust = 1),
           axis.title.y = element_blank())
dev.off()



