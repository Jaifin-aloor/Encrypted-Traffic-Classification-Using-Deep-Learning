import torch
import numpy as np
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict
import time
import os
import warnings
warnings.filterwarnings('ignore')

print("🔥 LOADING HYBRID MODEL FOR REAL LIVE TRAFFIC...")
print("="*60)

class HybridCNNLSTM(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(32, 64, 3, padding=1),
            torch.nn.ReLU()
        )
        self.lstm = torch.nn.LSTM(64, 128, batch_first=True, dropout=0.3)
        self.fc = torch.nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = HybridCNNLSTM(num_features=8, num_classes=2)

try:
    model.load_state_dict(torch.load('hybrid_model.pth', map_location=device))
    print("✅ Model loaded!")
except:
    print("❌ Run train.py first to generate hybrid_model.pth")
    exit()

model.to(device)
model.eval()

class LiveFlowAnalyzer:
    def __init__(self):
        self.flows = defaultdict(list)
        self.start_time = time.time()
        
    def process_packet(self, pkt):
        if IP not in pkt:
            return
            
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        proto = pkt[IP].proto
        
        if TCP in pkt:
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
        elif UDP in pkt:
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
        else:
            return
            
        flow_key = (src_ip, sport, dst_ip, dport, proto)
        reverse_key = (dst_ip, dport, src_ip, sport, proto)
        
        pkt_info = {
            'time': float(pkt.time),
            'len': len(pkt),
            'src_ip': src_ip,
            'dst_ip': dst_ip
        }
        
        self.flows[flow_key].append(pkt_info)
        self.flows[reverse_key].append(pkt_info)
    
    def extract_features(self, flow_packets):
        if len(flow_packets) < 2:
            return None
            
        packets = sorted(flow_packets, key=lambda x: x['time'])
        duration = packets[-1]['time'] - packets[0]['time']
        
        fwd_pkts = [p for p in packets[:len(packets)//2]]
        bwd_pkts = [p for p in packets[len(packets)//2:]]
        
        features = [
            duration,                    # 1. Duration
            len(fwd_pkts),              # 2. Fwd packets  
            len(bwd_pkts),              # 3. Bwd packets
            sum(p['len'] for p in fwd_pkts),  # 4. Fwd length
            sum(p['len'] for p in bwd_pkts),  # 5. Bwd length
        ]
        
        fwd_times = sorted([p['time'] for p in fwd_pkts])
        if len(fwd_times) > 1:
            iats = [fwd_times[i+1] - fwd_times[i] for i in range(len(fwd_times)-1)]
            features.extend([min(iats), max(iats), np.mean(iats)])  # 6,7,8 IATs
        else:
            features.extend([0, 0, 0])
            
        return np.array(features)
    
    def get_flows_for_analysis(self):
        features_list = []
        for flow_key, packets in self.flows.items():
            if len(packets) >= 3:
                feats = self.extract_features(packets)
                if feats is not None:
                    features_list.append((flow_key, feats))
        self.flows.clear()
        return features_list

analyzer = LiveFlowAnalyzer()

def packet_callback(pkt):
    analyzer.process_packet(pkt)

print(f"✅ Live capture started on WiFi...")
print(f"{'Time':<10} {'Flows':<8} {'VPN':<6} {'Normal':<8} {'Sample IPs'}")
print("-"*60)

vpn_count = 0
normal_count = 0
total_flows = 0

try:
    while True:
        # Capture for 3 seconds
        sniff(prn=packet_callback, timeout=3, store=0)
        
        flows = analyzer.get_flows_for_analysis()
        if flows:
            total_flows += len(flows)
            
            # Process flows
            batch_features = []
            for _, feats in flows:
                feats_norm = (feats - feats.mean()) / (feats.std() + 1e-6)
                batch_features.append(feats_norm)
            
            if batch_features:
                x = torch.FloatTensor(np.array(batch_features)).unsqueeze(1).to(device)
                
                with torch.no_grad():
                    outputs = model(x)
                    probs = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(probs, dim=1).cpu().numpy()
                
                batch_vpn = np.sum(predictions == 1)
                batch_normal = len(predictions) - batch_vpn
                
                vpn_count += batch_vpn
                normal_count += batch_normal
                
                # Show sample IPs
                sample_ip = flows[0][0][0] if flows else "N/A"
                
                print(f"{time.strftime('%H:%M:%S'):<10} "
                      f"{len(flows):<8} "
                      f"{batch_vpn:<6} "
                      f"{batch_normal:<8} "
                      f"{sample_ip[:15]}...")
        
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n\n" + "="*60)
    print("📊 LIVE TRAFFIC ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total Flows Analyzed:  {total_flows}")
    print(f"🔴 VPN/Encrypted Detected: {vpn_count}")
    print(f"🟢 Normal Traffic:       {normal_count}")
    print(f"VPN Detection Rate:     {vpn_count/(vpn_count+normal_count)*100:.1f}%" if (vpn_count+normal_count)>0 else "N/A")
    print("\n✅ REAL LIVE DEMO COMPLETE!")
