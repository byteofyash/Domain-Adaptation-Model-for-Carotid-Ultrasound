import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ==========================================
# 1. GENERATOR ARCHITECTURE
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Initial Convolution (feat1)
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, padding=0)
        self.in1 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Downsampling 1 (feat2)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.in2 = nn.InstanceNorm2d(128)
        
        # Downsampling 2 (feat3)
        self.pad3 = nn.ReflectionPad2d(1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0)
        self.in3 = nn.InstanceNorm2d(256)
        
        # Extra Conv Block (to match paper's 4th block)
        self.pad4 = nn.ReflectionPad2d(1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.in4 = nn.InstanceNorm2d(256)
        
        # 9 Residual Blocks
        self.resblocks = nn.Sequential(*[ResidualBlock(256) for _ in range(9)])
        
        # Upsampling 1
        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.in_d1 = nn.InstanceNorm2d(256)
        
        # Upsampling 2
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in_d2 = nn.InstanceNorm2d(128)
        
        # Upsampling 3
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in_d3 = nn.InstanceNorm2d(64)
        
        # Final Feature Extractor (last_feat)
        self.pad5 = nn.ReflectionPad2d(3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=7, padding=0)
        self.in5 = nn.InstanceNorm2d(64)
        
        # Output Layer
        self.final_conv = nn.Conv2d(64, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # Forward pass tracking specific layers for style/noise and content loss
        x = self.pad1(x)
        feat1 = self.relu(self.in1(self.conv1(x)))
        
        x = self.pad2(feat1)
        feat2 = self.relu(self.in2(self.conv2(x)))
        
        x = self.pad3(feat2)
        feat3 = self.relu(self.in3(self.conv3(x)))
        
        x = self.pad4(feat3)
        x = self.relu(self.in4(self.conv4(x)))
        
        x = self.resblocks(x)
        
        x = self.relu(self.in_d1(self.deconv1(x)))
        x = self.relu(self.in_d2(self.deconv2(x)))
        x = self.relu(self.in_d3(self.deconv3(x)))
        
        x = self.pad5(x)
        last_feat = self.relu(self.in5(self.conv5(x)))
        
        out = torch.sigmoid(self.final_conv(last_feat))
        
        return out, feat1, feat2, feat3, last_feat

# ==========================================
# 2. DISCRIMINATOR ARCHITECTURE
# ==========================================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Input: (1, 400, 400) -> Output: (1)
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(),
            nn.Linear(512 * 25 * 25, 1) # Matches the flattened shape of a 400x400 input
        )

    def forward(self, x):
        return self.model(x)

# ==========================================
# 3. MATHEMATICAL UTILITIES (LOSSES)
# ==========================================
def gram_matrix(x):
    """Calculates the Gram Matrix for a given feature map."""
    b, c, h, w = x.size()
    features = x.view(b, c, h * w)
    # Batch matrix multiplication: features * features^T
    gram = torch.bmm(features, features.transpose(1, 2))
    # Normalize by total number of elements
    return gram / (c * h * w)

def calculate_noise_loss(g_feats, r_feats):
    """Computes the MSE between the Gram matrices of the early layers."""
    mse_loss = nn.MSELoss()
    loss = 0.0
    for g, r in zip(g_feats, r_feats):
        loss += mse_loss(gram_matrix(g), gram_matrix(r))
    return loss

# ==========================================
# 4. TRAINING LOOP SETUP
# ==========================================
def train_domain_adaptation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Networks
    G = Generator().to(device)
    D_c = Discriminator().to(device) # Content Discriminator
    D_n = Discriminator().to(device) # Noise/Style Discriminator
    
    # 2. Optimizers
    lr = 0.0002
    beta1 = 0.5
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_D_c = optim.Adam(D_c.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_D_n = optim.Adam(D_n.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # 3. Learning Rate Schedulers (Linear decay from epoch 100 to 200)
    def lr_lambda(epoch):
        return 1.0 - max(0, epoch - 100) / 100.0
    
    sch_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lr_lambda)
    sch_D_c = optim.lr_scheduler.LambdaLR(opt_D_c, lr_lambda=lr_lambda)
    sch_D_n = optim.lr_scheduler.LambdaLR(opt_D_n, lr_lambda=lr_lambda)
    
    # 4. Objective Functions
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    lambda_content = 10.0
    lambda_noise = 10.0
    
    # 5. Scaler for Mixed Precision
    scaler = torch.amp.GradScaler('cuda')

    epochs = 200
    
    # NOTE: Replace 'train_loader' with your actual PyTorch DataLoader mapping to Domain A and B
    # train_loader = ... 
    
    print("Starting Training Loop...")
    for epoch in range(epochs):
        # for batch_idx, (real_x, real_y) in enumerate(train_loader):
            
            # --- Assuming real_x (Source) and real_y (Target) are loaded onto device ---
            # real_x = real_x.to(device)
            # real_y = real_y.to(device)
            
            # ==========================================
            # A. DISCRIMINATOR UPDATES (D_c & D_n)
            # ==========================================
            opt_D_c.zero_grad()
            opt_D_n.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # 1. Forward Pass Generator
                fake_y, _, _, _, _ = G(real_x)
                
                # 2. Content Discriminator Loss (D_c)
                dc_r_logits = D_c(real_x)
                dc_f_logits = D_c(fake_y.detach()) # Detach fake_y to avoid computing G gradients
                
                loss_dc_real = bce_loss(dc_r_logits, torch.ones_like(dc_r_logits))
                loss_dc_fake = bce_loss(dc_f_logits, torch.zeros_like(dc_f_logits))
                loss_D_c = loss_dc_real + loss_dc_fake
                
                # 3. Noise Discriminator Loss (D_n)
                dn_r_logits = D_n(real_y)
                dn_f_logits = D_n(fake_y.detach())
                
                loss_dn_real = bce_loss(dn_r_logits, torch.ones_like(dn_r_logits))
                loss_dn_fake = bce_loss(dn_f_logits, torch.zeros_like(dn_f_logits))
                loss_D_n = loss_dn_real + loss_dn_fake
                
            # Backprop D_c
            scaler.scale(loss_D_c).backward()
            scaler.step(opt_D_c)
            
            # Backprop D_n
            scaler.scale(loss_D_n).backward()
            scaler.step(opt_D_n)

            # ==========================================
            # B. GENERATOR UPDATE (G)
            # ==========================================
            opt_G.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # 1. Forward Pass real_x to get fakes and features
                fake_y, g_feat1, g_feat2, g_feat3, _ = G(real_x)
                
                # 2. Forward pass real_x AGAIN just to get its last feature map for Content Loss
                # Alternatively, you can save real_x's last feature map earlier to save computation
                _, _, _, _, rx_last_feat = G(real_x)
                
                # 3. Forward pass fake_y through G to get its last feature map for Content Loss
                _, _, _, _, fake_last_feat = G(fake_y)
                
                # 4. Forward pass real_y through G to extract its target style/noise features
                with torch.no_grad():
                    _, ry_f1, ry_f2, ry_f3, _ = G(real_y)
                
                # --- Calculate Generator Objective Losses ---
                
                # Adversarial Loss (Fooling D_c and D_n)
                dc_f_logits_for_G = D_c(fake_y)
                dn_f_logits_for_G = D_n(fake_y)
                
                adv_g_loss = bce_loss(dc_f_logits_for_G, torch.ones_like(dc_f_logits_for_G)) + \
                             bce_loss(dn_f_logits_for_G, torch.ones_like(dn_f_logits_for_G))
                
                # Content Loss (L1 difference between features of x and G(x))
                loss_content = l1_loss(fake_last_feat, rx_last_feat)
                
                # Style/Noise Loss (Wasserstein approximation via Gram matrices)
                loss_noise = calculate_noise_loss(
                    g_feats=[g_feat1, g_feat2, g_feat3], 
                    r_feats=[ry_f1, ry_f2, ry_f3]
                )
                
                # Total Generator Loss
                total_g_loss = adv_g_loss + (lambda_content * loss_content) + (lambda_noise * loss_noise)

            # Backprop G
            scaler.scale(total_g_loss).backward()
            scaler.step(opt_G)
            
            # Update Scaler for next iteration
            scaler.update()

        # Step the learning rate schedulers at the end of the epoch
        # sch_G.step()
        # sch_D_c.step()
        # sch_D_n.step()
        
        # print(f"Epoch [{epoch+1}/{epochs}] | G Loss: {total_g_loss.item():.4f} | D_c Loss: {loss_D_c.item():.4f} | D_n Loss: {loss_D_n.item():.4f}")

if __name__ == '__main__':
    # Uncomment to test memory allocation and trace execution
    # train_domain_adaptation()
    pass
