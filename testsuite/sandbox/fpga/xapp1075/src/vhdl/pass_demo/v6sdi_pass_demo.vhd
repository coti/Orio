-------------------------------------------------------------------------------- 
--   ____  ____ 
--  /   /\/   / 
-- /___/  \  /   Vendor: Xilinx 
-- \   \   \/    Author: John F. Snow
--  \   \        Filename: $RCSfile: v6sdi_pass_demo.vhd,rcs $
--  /   /        Date Last Modified:  $Date: 2010-10-20 10:24:50-06 $
-- /___/   /\    Date Created: January 29, 2010
-- \   \  /  \ 
--  \___\/\___\ 
-- 
--
-- Revision History: 
-- $Log: v6sdi_pass_demo.vhd,rcs $
-- Revision 1.2  2010-10-20 10:24:50-06  jsnow
-- DRPCLK was changed from the SystemACE clock to the 27 MHz clock
-- from the FMC because different versions of the ML605 board have
-- different SystemACE clock frequencies. The v6gtx_sdi_control instantiation
-- was updated to match changes to this module.
--
-- Revision 1.1  2010-03-11 17:19:53-07  jsnow
-- The GTX wrapper was modified to be compatible with the way the
-- ISE 12.1 RocketIO wizard will create GTX wrappers when using
-- the HDSDI protocol template. Modified the v6gtx_sdi_control module
-- to replace the txpll_div_rst port with a 13-bit GTXTEST port.
--
-- Revision 1.0  2010-03-08 14:09:41-07  jsnow
-- Initial release.
--
-------------------------------------------------------------------------------- 
--   
-- (c) Copyright 2010 Xilinx, Inc. All rights reserved.
-- 
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
-- 
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of,
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
-- 
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
-- 
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES. 
--
-------------------------------------------------------------------------------- 
--
-- Module Description:
--
--This demo implements a one-channel, triple-rate SDI pass-through interface for
--the ML605 board and the broadcast FMC mezzanine card. A Si5324 clock module
--must be installed in the clock module "L" slot of the FMC mezzanine card. One
--Si5324 device on this module is used to remove jitter from RXRECCLK in HD/3G
--modes to produce a clean 148.5 or 148.35 MHz reference clock for the TX. In
--SD mode, the 27 MHz data ready signal from the DRU is sent to the Si5324 where
--jitter is removed and it is multiplied up to 148.5 MHz to be used as the TX
--reference clock.
--
--The active SDI RX and TX are RX2 and TX2. 
--
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.hdsdi_pkg.all;
use work.anc_edh_pkg.all;

library unisim; 
use unisim.vcomponents.all; 

entity v6sdi_pass_demo is
port (
-- MGTs
    FMC_HPC_DP1_C2M_N:          out std_logic;
    FMC_HPC_DP1_C2M_P:          out std_logic;
    FMC_HPC_DP1_M2C_N:          in  std_logic;
    FMC_HPC_DP1_M2C_P:          in  std_logic;

-- MGT REFCLKs
    FMC_HPC_GBTCLK0_M2C_N:      in  std_logic;      -- Q113 MGTREFCLK0 (148.5 MHz RX refclk)
    FMC_HPC_GBTCLK0_M2C_P:      in  std_logic;
    FMC_HPC_CLK2_M2C_MGT_C_N:   in  std_logic;      -- Q113 MGTREFCLK1 (148.X MHz TX refclk)
    FMC_HPC_CLK2_M2C_MGT_C_P:   in  std_logic;

-- AVB FMC card connections
    FMC_HPC_HB06_CC_N:          in  std_logic;      -- 27 MHz Clock from FMC
    FMC_HPC_HB06_CC_P:          in  std_logic;
    
    FMC_HPC_LA00_CC_P:          out std_logic;      -- main SPI interface SCK
    FMC_HPC_LA00_CC_N:          out std_logic;      -- main SPI interface MOSI
    FMC_HPC_LA27_P:             in  std_logic;      -- main SPI interface MISO
    FMC_HPC_LA14_N:             out std_logic;      -- mian SPI interface SS

-- Clock Module L connections
    FMC_HPC_LA29_P:             out std_logic;      -- CML SPI interface SCK
    FMC_HPC_LA29_N:             in  std_logic;      -- CML SPI interface MISO
    FMC_HPC_LA07_P:             out std_logic;      -- CML SPI interface MOSI
    FMC_HPC_LA07_N:             out std_logic;      -- CML SPI interface SS

    FMC_HPC_LA13_P:             out std_logic;      -- Si5324 A reset asserted low
    FMC_HPC_LA33_P:             out std_logic;      -- Si5324 B reset asserted low
    FMC_HPC_LA26_P:             out std_logic;      -- Si5324 C reset asserted low
    
    FMC_HPC_LA16_P:             out std_logic;      -- Output RXRECCLK (HD/3G modes) or
    FMC_HPC_LA16_N:             out std_logic;      -- SD-SDI DRU ready to CML Si5324 B CKIN2

-- Misc ML605 IO
    GPIO_LED_0:                 out std_logic;      -- RX locked status is shown on this LED
    USER_SMA_GPIO_P:            out std_logic;
    USER_SMA_GPIO_N:            out std_logic;
    LCD_DB4:                    out std_logic;
    LCD_DB5:                    out std_logic;
    LCD_DB6:                    out std_logic;
    LCD_DB7:                    out std_logic;
    LCD_E:                      out std_logic;
    LCD_RW:                     out std_logic;
    LCD_RS:                     out std_logic;
    GPIO_SW_C:                  in  std_logic;
    GPIO_SW_W:                  in  std_logic;
    GPIO_SW_E:                  in  std_logic;
    GPIO_SW_N:                  in  std_logic;
    GPIO_SW_S:                  in  std_logic);
end v6sdi_pass_demo;

architecture xilinx of v6sdi_pass_demo is

attribute equivalent_register_removal : string;
attribute keep : string;

--------------------------------------------------------------------------------
-- Internal signals definitions

-- Reference clocks
signal drpclk :             std_logic;                                      -- 27 MHz DRP clock
signal mgtclk_rx :          std_logic;                                      -- RX MGT reference clock
signal mgtclk_tx :          std_logic;                                      -- TX MGT reference clock
signal mgtrefclkrx :        std_logic_vector(1 downto 0);
signal mgtrefclktx :        std_logic_vector(1 downto 0);
signal clk_fmc_27M :        std_logic;                                      -- 27 MHz clock for FMC interfaces (regional)
signal clk_fmc_27M_in :     std_logic;
signal Si5324_b_clkin :     std_logic;                                      -- input clock to Si5324 B on clock module L

signal gtxtxreset :         std_logic;                                      -- GTXTXRESET
signal LOL_sync :           std_logic_vector(1 downto 0) := "00";           -- Syncronizes Si53243 B LOL to drpclk
signal gtxtxreset_dly :     unsigned(9 downto 0) := "0000000000";           -- Generates short delay to hold GTXTXRESET after fall of Si5324 LOL
signal gtxtxreset_dly_tc :  std_logic;                                      -- Terminal count for GTXTXRESET delay counter

-- TX2 signals
signal tx2_outclk :         std_logic;                                      -- TXOUTCLK from TX2 MGT
signal tx2_usrclk :         std_logic;                                      -- Driven by RXRECCLK for HD/3G and TXOUTCLK for SD
signal tx2_pll_locked :     std_logic;                                      -- 1 = TX PMA PLL locked to reference clock
signal tx2_gtx_data :       std_logic_vector(19 downto 0);                  -- Encoded data stream to MGT TX
signal tx2_reset :          std_logic;                                      -- Connects to MGT TXRESET input
signal tx2_bufstatus :      std_logic_vector(1 downto 0);                   -- MGT TXBUFSTATUS output
signal tx2_resetdone :      std_logic;                                      -- 1 = MGT TX reset done
signal tx2_gtxtest:         std_logic_vector(12 downto 0);                  -- GTXTEST
signal tx2_din_rdy :        std_logic;                                      -- TX clock enable used in 3G-SDI level B mode
signal tx2_mode :           std_logic_vector(1 downto 0) := "00";           -- TX SDI mode: 00=HD, 01=SD, 10=3G
signal tx2_out_mode :       std_logic_vector(1 downto 0) := "00";           -- Controls the operating mode of the TX output module
signal tx2_ds1a :           std_logic_vector(9 downto 0) := (others => '0');-- Link A data stream 1
signal tx2_ds2a :           std_logic_vector(9 downto 0) := (others => '0');-- Link A data stream 2
signal tx2_ds1b :           std_logic_vector(9 downto 0) := (others => '0');-- Link B data stream 1
signal tx2_ds2b :           std_logic_vector(9 downto 0) := (others => '0');-- Link B data stream 2
signal tx2_eav :            std_logic := '0';                               -- TX EAV
signal tx2_sav :            std_logic := '0';                               -- TX SAV
signal fifo2_dout :         std_logic_vector(9 downto 0);                   -- SD FIFO output
signal fifo2_syn :          std_logic_vector(1 downto 0) := "00";           -- Synchronizer for SD FIFO half-full signal
signal fifo2_half_syn :     std_logic;                                      -- SD FIFO half-full signal, synchronized to tx2_usrclk
signal fifo2_half_full:     std_logic;                                      -- SD FIFO half-full output, synchronized to rx2_usrclk
signal fifo2_empty :        std_logic;                                      -- SD FIFO empty output
signal fifo2_full :         std_logic;                                      -- SD FIFO full output
signal fifo2_rden :         std_logic := '0';                               -- SD FIFO read enable
signal tx2_slew :           std_logic;                                      -- Cable driver slew rate control
signal tx2_ce :             std_logic_vector(1 downto 0) := "00";           -- TX clock enable for SD mode
signal tx2_ce_gen :         std_logic_vector(10 downto 0) := "00000100001"; -- Generates 5/6/5/6 cadence for TX SD clock enable
signal tx2_ce_mux :         std_logic;                                      -- Used to generate tx2_ce
signal tx2_postemphasis :   std_logic_vector(4 downto 0);                   -- GTX TX postemphasis setting
signal tx2_mode_int :       std_logic_vector(1 downto 0) := "00";           -- tx2_mode generated in rx2_usrclk domain
signal tx2_mode_sync1 :     std_logic_vector(1 downto 0) := "00";           -- Used to sync tx2_mode_int to FMC clock domain
signal tx2_mode_sync2 :     std_logic_vector(1 downto 0) := "00";           -- This version of tx2_mode is synchronous with clk_fmc_27M

attribute equivalent_register_removal of tx2_ce : signal is "no";
attribute keep of tx2_ce : signal is "TRUE";

-- RX2 signals
signal rx2_recclk :         std_logic;                                      -- MGT RXRECCLK
signal rx2_clr_errs :       std_logic;                                      -- User control to clear RX CRC & EDH error counts
signal rx2_usrclk :         std_logic;                                      -- Global RXUSRCLK2
signal rx2_mode :           std_logic_vector(1 downto 0);                   -- RX SDI mode: 00=HD, 01=SD, 10=3G
signal rx2_mode_HD :        std_logic;                                      -- 1 = RX SDI mode = HD
signal rx2_mode_SD :        std_logic;                                      -- 1 = RX SDI mode = SD
signal rx2_mode_3G :        std_logic;                                      -- 1 = RX SDI mode = 3G
signal rx2_mode_SD_syncer : std_logic_vector(1 downto 0) := "00";           -- used to sync rx2_mode_SD to tx2_usrclk
signal rx2_mode_SD_sync :   std_logic;                                      -- rx2_mode_SD synced to tx2_usrclk
signal rx2_mode_3G_syncer : std_logic_vector(1 downto 0) := "00";           -- used to sync rx2_mode_3G to tx2_usrclk
signal rx2_mode_3G_sync :   std_logic;                                      -- rx2_mode_3G synced to tx2_usrclk
signal rx2_mode_locked:     std_logic;                                      -- 1 = RX SDI mode detection locked
signal rx2_hd_locked :      std_logic;                                      -- 1 = RX HD/3G format detector locked 
signal rx2_sd_locked :      std_logic;                                      -- 1 = RX SD format detector locked
signal rx2_locked :         std_logic;                                      -- 1 = RX format detector locked
signal rx2_hd_format :      std_logic_vector(3 downto 0);                   -- HD/3G transport format
signal rx2_sd_format :      vidstd_type;                                    -- SD transport format
signal rx2_format :         std_logic_vector(3 downto 0);                   -- Incoming signal transport format
signal rx2_level_b :        std_logic;                                      -- 1 = input signal is 3G-SDI level B
signal rx2_level_b_syncer : std_logic_vector(1 downto 0) := "00";           -- used to sync rx2_level_b to tx2_usrclk
signal rx2_level_b_sync :   std_logic;                                      -- rx2_level_b synced to tx2_usrclk
signal rx2_m :              std_logic;                                      -- 1 = input bit rate is 1/1.001
signal rx2_ce :             std_logic_vector(2 downto 0);                   -- RX SD clock enable
signal rx2_dout_rdy_3G :    std_logic_vector(0 downto 0);                   -- RX 3G-B clock enable
signal rx2_ln_a :           xavb_hd_line_num_type;                          -- RX line number link A
signal rx2_ln_b :           xavb_hd_line_num_type;                          -- RX line number link B
signal rx2_a_vpid :         std_logic_vector(31 downto 0);                  -- SMPTE 352 video payload ID data link A data stream
signal rx2_a_vpid_valid :   std_logic;                                      -- 1 = rx2_a_vpid valid
signal rx2_crc_err_a :      std_logic;                                      -- CRC error detected, link A data stream
signal rx2_crc_err_b :      std_logic;                                      -- CRC error detected, link B data stream (3G-B only)
signal rx2_crc_err :        std_logic;                                      -- CRC error detected, either data stream
signal rx2_crc_err_capture: std_logic := '0';                               -- FF used to capture & hold rx2_crc_err
signal rx2_b_vpid :         std_logic_vector(31 downto 0);                  -- SMPTE 352 video payload ID data link B data stream
signal rx2_b_vpid_valid :   std_logic;                                      -- 1 = rx2_b_vpid_valid
signal rx2_ds1a :           xavb_data_stream_type;                          -- Link A data stream 1
signal rx2_ds2a :           xavb_data_stream_type;                          -- Link A data stream 2
signal rx2_ds1b :           xavb_data_stream_type;                          -- Link B data stream 1 (3G-B only)
signal rx2_ds2b :           xavb_data_stream_type;                          -- Link B data stream 2 (3G-B only)
signal rx2_eav :            std_logic;                                      -- RX EAV
signal rx2_sav :            std_logic;                                      -- RX SAV
signal rx2_edh_errcnt :     std_logic_vector(23 downto 0);                  -- 1 = SD EDH error count
signal rx2_rate :           std_logic_vector(1 downto 0);                   -- Control MGT RXRATE port
signal rx2_bufstatus :      std_logic_vector(2 downto 0);                   -- MGT RXBUFSTATUS
signal rx2_bufreset :       std_logic;                                      -- MGT RXBUFRESET
signal rx2_cdrreset :       std_logic;                                      -- MGT RXCDRRESET
signal rx2_gtxreset :       std_logic;                                      -- MGT RXGTXRESET
signal rx2_pll_locked :     std_logic;                                      -- MGT RX PMA PLL locked to reference clock
signal rx2_ratedone :       std_logic;                                      -- MGT RXRATEDONE
signal rx2_resetdone :      std_logic;                                      -- MGT RXRESETDONE
signal rx2_gtx_data :       std_logic_vector(19 downto 0);                  -- Raw received data from MGT RX
signal Si5324_B_out_fsel :  std_logic_vector(3 downto 0) := "0000";         -- Selects output frequency for Si5324 B on clock module L
signal Si5324_B_in_fsel :   std_logic_vector(4 downto 0) := "00000";        -- Selects input frequency for Si5324 B on clock module L
signal Si5324_B_bw_sel :    std_logic_vector(3 downto 0) := "0000";         -- Selects the bandwidth for Si5324 B on clock module L
signal rx2_m_sync :         std_logic_vector(1 downto 0) := "00";           -- Synchronizes rx2_m to TXUSRCLK2
signal rx2_edh_rst :        std_logic;
signal rx2_mode_change_gen: std_logic_vector(1 downto 0) := "00";           -- Used to detect change in rx2_mode
signal rx2_mode_locked_rise: std_logic := '0';                              -- 1 on rising edge of rx2_mode_locked
signal rx2_mode_locked_rise_txsync : std_logic_vector(2 downto 0) := "000"; -- Synchronizes rx2_mode_locked_rise to tx2_usrclk

-- MGT DRP signals
signal dp1_do :             std_logic_vector(15 downto 0);
signal dp1_drdy :           std_logic;
signal dp1_daddr :          std_logic_vector(7 downto 0);
signal dp1_di :             std_logic_vector(15 downto 0);
signal dp1_den :            std_logic;
signal dp1_dwe :            std_logic;

-- AVB FMC mezzanine card signals 
signal fmc_tx2_red_led :    std_logic_vector(1 downto 0);                   -- Controls TX2 red LED on FMC card
signal fmc_tx2_grn_led :    std_logic_vector(1 downto 0);                   -- Controls TX2 green LED on FMC card
signal fmc_sdi_eq_cd_n :    std_logic_vector(7 downto 0);                   -- Carrier detect signals from SDI cable EQs
signal fmc_sdi_eq_cli :     std_logic_vector(4 downto 0);                   -- Cable length indicator signals from SDI cable EQs
signal fmc_sdi_drv_hd_sd :  std_logic_vector(7 downto 0);                   -- Slew reate control for SDI cable drivers
signal fmc_sdi_drv_enable : std_logic_vector(7 downto 0);                   -- SDI cable driver enables
signal fmc_sdi_drv_fault_n: std_logic_vector(7 downto 0);                   -- SDI cable driver fault signals
signal fmc_fpga_rev :       std_logic_vector(7 downto 0);                   -- FMC FPGA revision
signal fmc_exp_brd_prsnt :  std_logic;                                      -- FMC expansion board present
signal fmc_Si5324_LOL :     std_logic;                                      -- Loss-of-lock signal from main Si5324 on FMC card
signal cml_Si5324_B_LOL :   std_logic;                                      -- Loss-of-lock signal from Si5324 B on clock module L
signal cml_type :           std_logic_vector(15 downto 0);
signal cml_type_valid :     std_logic;
signal cml_type_error :     std_logic;
signal lcd_d :              std_logic_vector(3 downto 0);                   -- LCD display data bus

-- ChipScope signals
signal control0 :           std_logic_vector(35 downto 0);
signal control1 :           std_logic_vector(35 downto 0);
signal control2 :           std_logic_vector(35 downto 0);
signal async_in :           std_logic_vector(51 downto 0);
signal rx_trig0 :           std_logic_vector(44 downto 0);
signal tx_trig0 :           std_logic_vector(58 downto 0);
signal sync_out :           std_logic_vector(7 downto 0);

component triple_sdi_rx_20b
generic (
    NUM_SD_CE:          integer := 2;
    NUM_3G_DRDY:        integer := 2;
    ERRCNT_WIDTH:       integer := 4;
    MAX_ERRS_LOCKED:    integer := 15;
    MAX_ERRS_UNLOCKED:  integer := 2);
port (
    -- inputs
    clk:            in  std_logic;
    rst:            in  std_logic;
    data_in:        in  std_logic_vector(19 downto 0);
    frame_en:       in  std_logic;

    -- general outputs
    mode:           out std_logic_vector(1 downto 0);
    mode_HD:        out std_logic;
    mode_SD:        out std_logic;
    mode_3G:        out std_logic;
    mode_locked:    out std_logic;
    rx_locked:      out std_logic;
    t_format:       out xavb_vid_format_type;
    level_b_3G:     out std_logic;
    ce_sd:          out std_logic_vector(NUM_SD_CE-1 downto 0);
    nsp:            out std_logic;
    ln_a:           out xavb_hd_line_num_type;
    a_vpid:         out std_logic_vector(31 downto 0);
    a_vpid_valid:   out std_logic;
    b_vpid:         out std_logic_vector(31 downto 0);
    b_vpid_valid:   out std_logic;
    crc_err_a:      out std_logic;
    ds1_a:          out xavb_data_stream_type;
    ds2_a:          out xavb_data_stream_type;
    eav:            out std_logic;
    sav:            out std_logic;
    trs:            out std_logic;

    -- outputs valid for 3G level B only
    ln_b:           out xavb_hd_line_num_type;
    dout_rdy_3G:    out std_logic_vector(NUM_3G_DRDY-1 downto 0);
    crc_err_b:      out std_logic;
    ds1_b:          out xavb_data_stream_type;
    ds2_b:          out xavb_data_stream_type;

    recclk_txdata:  out std_logic_vector(19 downto 0));
end component;

component edh_processor
port (
    clk:            in  std_ulogic;
    ce:             in  std_ulogic;
    rst:            in  std_ulogic;

    -- video decoder inputs
    vid_in:         in  video_type;
    reacquire:      in  std_ulogic;
    en_sync_switch: in  std_ulogic;
    en_trs_blank:   in  std_ulogic;

    -- EDH flag inputs
    anc_idh_local:  in  std_ulogic;
    anc_ues_local:  in  std_ulogic;
    ap_idh_local:   in  std_ulogic;
    ff_idh_local:   in  std_ulogic;
    errcnt_flg_en:  in  edh_allflg_type;
    clr_errcnt:     in  std_ulogic;
    receive_mode:   in  std_ulogic;

    -- video and decoded video timing outputs
    vid_out:        out video_type;
    std:            out vidstd_type;
    std_locked:     out std_ulogic;
    trs:            out std_ulogic;
    field:          out std_ulogic;
    v_blank:        out std_ulogic;
    h_blank:        out std_ulogic;
    horz_count:     out hpos_type;
    vert_count:     out vpos_type;
    sync_switch:    out std_ulogic;
    locked:         out std_ulogic;
    eav_next:       out std_ulogic;
    sav_next:       out std_ulogic;
    xyz_word:       out std_ulogic;
    anc_next:       out std_ulogic;
    edh_next:       out std_ulogic;

    -- EDH flag outputs
    rx_ap_flags:    out edh_flgset_type;
    rx_ff_flags:    out edh_flgset_type;
    rx_anc_flags:   out edh_flgset_type;
    ap_flags:       out edh_flgset_type;
    ff_flags:       out edh_flgset_type;
    anc_flags:      out edh_flgset_type;
    packet_flags:   out edh_pktflg_type;
    errcnt:         out edh_errcnt_type;
    edh_packet:     out std_ulogic);
end component;

component triple_sdi_tx_output_20b
port (
    clk:            in  std_logic;
    din_rdy:        in  std_logic;
    ce:             in  std_logic_vector(1 downto 0);
    rst:            in  std_logic;
    mode:           in  std_logic_vector(1 downto 0);
    ds1a:           in  xavb_data_stream_type;
    ds2a:           in  xavb_data_stream_type;
    ds1b:           in  xavb_data_stream_type;
    ds2b:           in  xavb_data_stream_type;
    insert_crc:     in  std_logic;
    insert_ln:      in  std_logic;
    insert_edh:     in  std_logic;
    ln_a:           in  xavb_hd_line_num_type;
    ln_b:           in  xavb_hd_line_num_type;
    eav:            in  std_logic;
    sav:            in  std_logic;
    txdata:         out std_logic_vector(19 downto 0);
    ce_align_err:   out std_logic);
end component;

component v6gtx_sdi_control
generic (
    PMA_RX_CFG_HD :         std_logic_vector(27 downto 0) := X"05ce055";
    PMA_RX_CFG_SD :         std_logic_vector(27 downto 0) := X"0f44000";
    PMA_RX_CFG_3G :         std_logic_vector(27 downto 0) := X"05CE055";
    DRPCLK_FREQ :           integer := 32000000;
    TX_PLL_OUT_DIV_HD :     integer := 2;
    TX_PLL_OUT_DIV_3G :     integer := 1;
    RX_PLL_OUT_DIV_HD :     integer := 2;
    RX_PLL_OUT_DIV_3G :     integer := 1);
port (
    drpclk :                in  std_logic;
    rst :                   in  std_logic;
    txusrclk :              in  std_logic;
    tx_mode :               in  std_logic_vector(1 downto 0);
    txreset_in :            in  std_logic;
    txresetdone :           in  std_logic;
    txbufstatus1 :          in  std_logic;
    txplllkdet :            in  std_logic;
    txreset_out :           out std_logic;
    gtxtest :               out std_logic_vector(12 downto 0);
    tx_rate_change_done :   out std_logic;
    tx_slew :               out std_logic;
    rxusrclk :              in  std_logic;
    rx_mode :               in  std_logic_vector(1 downto 0);
    rxresetdone :           in  std_logic;
    rxbufstatus2 :          in  std_logic;
    rxratedone :            in  std_logic;
    rxcdrreset :            out std_logic;
    rxbufreset :            out std_logic;
    rxrate :                out std_logic_vector(1 downto 0);
    rx_m :                  out std_logic;
    drpdo :                 in  std_logic_vector(15 downto 0);
    drdy :                  in  std_logic;
    daddr :                 out std_logic_vector(7 downto 0);
    di :                    out std_logic_vector(15 downto 0);
    den :                   out std_logic;
    dwe :                   out std_logic);
end component;

component V6SDI_WRAPPER
generic
(
    WRAPPER_SIM_GTXRESET_SPEEDUP    : integer   := 0); -- Set to 1 to speed up sim reset
port(
    GTX0_LOOPBACK_IN                        : in   std_logic_vector(2 downto 0);
    GTX0_RXDATA_OUT                         : out  std_logic_vector(19 downto 0);
    GTX0_RXRECCLK_OUT                       : out  std_logic;
    GTX0_RXRESET_IN                         : in   std_logic;
    GTX0_RXUSRCLK2_IN                       : in   std_logic;
    GTX0_RXCDRRESET_IN                      : in   std_logic;
    GTX0_RXN_IN                             : in   std_logic;
    GTX0_RXP_IN                             : in   std_logic;
    GTX0_RXBUFSTATUS_OUT                    : out  std_logic_vector(2 downto 0);
    GTX0_RXBUFRESET_IN                      : in   std_logic;
    GTX0_MGTREFCLKRX_IN                     : in   std_logic_vector(1 downto 0);
    GTX0_PERFCLKRX_IN                       : in   std_logic;
    GTX0_GREFCLKRX_IN                       : in   std_logic;
    GTX0_NORTHREFCLKRX_IN                   : in   std_logic_vector(1 downto 0);
    GTX0_SOUTHREFCLKRX_IN                   : in   std_logic_vector(1 downto 0);
    GTX0_RXPLLREFSELDY_IN                   : in   std_logic_vector(2 downto 0);
    GTX0_GTXRXRESET_IN                      : in   std_logic;
    GTX0_PLLRXRESET_IN                      : in   std_logic;
    GTX0_RXPLLLKDET_OUT                     : out  std_logic;
    GTX0_RXRATE_IN                          : in   std_logic_vector(1 downto 0);
    GTX0_RXRATEDONE_OUT                     : out  std_logic;
    GTX0_RXRESETDONE_OUT                    : out  std_logic;
    GTX0_DADDR_IN                           : in   std_logic_vector(7 downto 0);
    GTX0_DCLK_IN                            : in   std_logic;
    GTX0_DEN_IN                             : in   std_logic;
    GTX0_DI_IN                              : in   std_logic_vector(15 downto 0);
    GTX0_DRDY_OUT                           : out  std_logic;
    GTX0_DRPDO_OUT                          : out  std_logic_vector(15 downto 0);
    GTX0_DWE_IN                             : in   std_logic;
    GTX0_TXDATA_IN                          : in   std_logic_vector(19 downto 0);
    GTX0_TXOUTCLK_OUT                       : out  std_logic;
    GTX0_TXRESET_IN                         : in   std_logic;
    GTX0_TXUSRCLK2_IN                       : in   std_logic;
    GTX0_TXDIFFCTRL_IN                      : in   std_logic_vector(3 downto 0);
    GTX0_TXN_OUT                            : out  std_logic;
    GTX0_TXP_OUT                            : out  std_logic;
    GTX0_TXPOSTEMPHASIS_IN                  : in   std_logic_vector(4 downto 0);
    GTX0_TXPREEMPHASIS_IN                   : in   std_logic_vector(3 downto 0);
    GTX0_MGTREFCLKTX_IN                     : in   std_logic_vector(1 downto 0);
    GTX0_PERFCLKTX_IN                       : in   std_logic;
    GTX0_GREFCLKTX_IN                       : in   std_logic;
    GTX0_NORTHREFCLKTX_IN                   : in   std_logic_vector(1 downto 0);
    GTX0_SOUTHREFCLKTX_IN                   : in   std_logic_vector(1 downto 0);
    GTX0_TXPLLREFSELDY_IN                   : in   std_logic_vector(2 downto 0);
    GTX0_TXBUFSTATUS_OUT                    : out  std_logic_vector(1 downto 0);
    GTX0_GTXTEST_IN                         : in   std_logic_vector(12 downto 0);
    GTX0_GTXTXRESET_IN                      : in   std_logic;
    GTX0_PLLTXRESET_IN                      : in   std_logic;
    GTX0_TXPLLLKDET_OUT                     : out  std_logic;
    GTX0_TXRESETDONE_OUT                    : out  std_logic);
end component;

component main_avb_control
port (
    clk:                in  std_logic;
    rst:                in  std_logic;
    sck:                out std_logic;
    mosi:               out std_logic;
    miso:               in  std_logic;
    ss:                 out std_logic;
    fpga_rev:           out std_logic_vector(7 downto 0);
    exp_brd_prsnt:      out std_logic;
    board_options:      out std_logic_vector(7 downto 0);
    xbar1_out0_sel:     in  std_logic_vector(1 downto 0);
    xbar1_out1_sel:     in  std_logic_vector(1 downto 0);
    xbar1_out2_sel:     in  std_logic_vector(1 downto 0);
    xbar1_out3_sel:     in  std_logic_vector(1 downto 0);
    xbar2_out0_sel:     in  std_logic_vector(1 downto 0);
    xbar2_out1_sel:     in  std_logic_vector(1 downto 0);
    xbar2_out2_sel:     in  std_logic_vector(1 downto 0);
    xbar2_out3_sel:     in  std_logic_vector(1 downto 0);
    xbar3_out0_sel:     in  std_logic_vector(1 downto 0);
    xbar3_out1_sel:     in  std_logic_vector(1 downto 0);
    xbar3_out2_sel:     in  std_logic_vector(1 downto 0);
    xbar3_out3_sel:     in  std_logic_vector(1 downto 0);
    Si5324_reset:       in  std_logic;
    Si5324_clkin_sel:   in  std_logic_vector(1 downto 0);
    Si5324_out_fsel:    in  std_logic_vector(3 downto 0);
    Si5324_in_fsel:     in  std_logic_vector(4 downto 0);
    Si5324_bw_sel:      in  std_logic_vector(3 downto 0);
    Si5324_DHOLD:       in  std_logic;
    Si5324_FOS2:        out std_logic;
    Si5324_FOS1:        out std_logic;
    Si5324_LOL:         out std_logic;
    Si5324_reg_adr:     in  std_logic_vector(7 downto 0);
    Si5324_reg_wr_dat:  in  std_logic_vector(7 downto 0);
    Si5324_reg_rd_dat:  out std_logic_vector(7 downto 0);
    Si5324_reg_wr:      in  std_logic;
    Si5324_reg_rd:      in  std_logic;
    Si5324_reg_rdy:     out std_logic := '0';
    Si5324_error:       out std_logic := '0';
    sync_video_fmt:     out std_logic_vector(10 downto 0) := (others => '0');
    sync_updating:      out std_logic := '0';
    sync_frame_rate:    out std_logic_vector(2 downto 0) := (others => '0');
    sync_m:             out std_logic := '0';
    sync_err:           out std_logic := '0';
    sdi_rx1_led:        in  std_logic_vector(1 downto 0);
    sdi_rx2_led:        in  std_logic_vector(1 downto 0);
    sdi_rx3_led:        in  std_logic_vector(1 downto 0);
    sdi_rx4_led:        in  std_logic_vector(1 downto 0);
    sdi_rx5_led:        in  std_logic_vector(1 downto 0);
    sdi_rx6_led:        in  std_logic_vector(1 downto 0);
    sdi_rx7_led:        in  std_logic_vector(1 downto 0);
    sdi_rx8_led:        in  std_logic_vector(1 downto 0);
    sdi_tx1_red_led:    in  std_logic_vector(1 downto 0);
    sdi_tx1_grn_led:    in  std_logic_vector(1 downto 0);
    sdi_tx2_red_led:    in  std_logic_vector(1 downto 0);
    sdi_tx2_grn_led:    in  std_logic_vector(1 downto 0);
    sdi_tx3_red_led:    in  std_logic_vector(1 downto 0);
    sdi_tx3_grn_led:    in  std_logic_vector(1 downto 0);
    sdi_tx4_red_led:    in  std_logic_vector(1 downto 0);
    sdi_tx4_grn_led:    in  std_logic_vector(1 downto 0);
    sdi_tx5_red_led:    in  std_logic_vector(1 downto 0);
    sdi_tx5_grn_led:    in  std_logic_vector(1 downto 0);
    sdi_tx6_red_led:    in  std_logic_vector(1 downto 0);
    sdi_tx6_grn_led:    in  std_logic_vector(1 downto 0);
    sdi_tx7_red_led:    in  std_logic_vector(1 downto 0);
    sdi_tx7_grn_led:    in  std_logic_vector(1 downto 0);
    sdi_tx8_red_led:    in  std_logic_vector(1 downto 0);
    sdi_tx8_grn_led:    in  std_logic_vector(1 downto 0);
    aes_rx1_red_led:    in  std_logic_vector(1 downto 0);
    aes_rx1_grn_led:    in  std_logic_vector(1 downto 0);
    aes_rx2_red_led:    in  std_logic_vector(1 downto 0);
    aes_rx2_grn_led:    in  std_logic_vector(1 downto 0);
    aes_tx1_red_led:    in  std_logic_vector(1 downto 0);
    aes_tx1_grn_led:    in  std_logic_vector(1 downto 0);
    aes_tx2_red_led:    in  std_logic_vector(1 downto 0);
    aes_tx2_grn_led:    in  std_logic_vector(1 downto 0);
    madi_rx_red_led:    in  std_logic_vector(1 downto 0);
    madi_rx_grn_led:    in  std_logic_vector(1 downto 0);
    madi_tx_red_led:    in  std_logic_vector(1 downto 0);
    madi_tx_grn_led:    in  std_logic_vector(1 downto 0);
    sync_red_led:       in  std_logic_vector(1 downto 0);
    sync_grn_led:       in  std_logic_vector(1 downto 0);
    sdi_eq_cd_n:        out std_logic_vector(7 downto 0) := (others => '0');
    sdi_eq_ext_3G_reach:in  std_logic_vector(7 downto 0);
    sdi_eq_select:      in  std_logic_vector(2 downto 0);
    sdi_eq_cli:         out std_logic_vector(4 downto 0) := (others => '0');
    sdi_drv_hd_sd:      in  std_logic_vector(7 downto 0);
    sdi_drv_enable:     in  std_logic_vector(7 downto 0);
    sdi_drv_fault_n:    out std_logic_vector(7 downto 0) := (others => '0'));
end component;

component cm_avb_control
port (
    clk:                in  std_logic;
    rst:                in  std_logic;
    ga:                 in  std_logic;
    sck:                out std_logic;
    mosi:               out std_logic;
    miso:               in  std_logic;
    ss:                 out std_logic;
    module_type:        out std_logic_vector(15 downto 0) := (others => '0');
    module_rev:         out std_logic_vector(15 downto 0) := (others => '0');
    module_type_valid:  out std_logic := '0';
    module_type_error:  out std_logic := '0';
    clkin5_src_sel:     in  std_logic;
    gpio_dir_0:         in  std_logic_vector(7 downto 0);
    gpio_dir_1:         in  std_logic_vector(7 downto 0);
    gpio_dir_2:         in  std_logic_vector(7 downto 0);
    gp_out_value_0:     in  std_logic_vector(7 downto 0);
    gp_out_value_1:     in  std_logic_vector(7 downto 0);
    gp_out_value_2:     in  std_logic_vector(7 downto 0);
    gp_in_value_0:      out std_logic_vector(7 downto 0) := (others => '0');
    gp_in_value_1:      out std_logic_vector(7 downto 0) := (others => '0');
    gp_in_value_2:      out std_logic_vector(7 downto 0) := (others => '0');
    gp_in:              out std_logic_vector(3 downto 0) := (others => '0');
    i2c_slave_adr:      in  std_logic_vector(7 downto 0);
    i2c_reg_adr:        in  std_logic_vector(7 downto 0);
    i2c_reg_dat_wr:     in  std_logic_vector(7 downto 0);
    i2c_reg_wr:         in  std_logic;
    i2c_reg_rd:         in  std_logic;
    i2c_reg_dat_rd:     out std_logic_vector(7 downto 0) := (others => '0');
    i2c_reg_rdy:        out std_logic := '0';
    i2c_reg_error:      out std_logic := '0';
    Si5324_A_clkin_sel: in  std_logic;
    Si5324_A_out_fsel:  in  std_logic_vector(3 downto 0);
    Si5324_A_in_fsel:   in  std_logic_vector(4 downto 0);
    Si5324_A_bw_sel:    in  std_logic_vector(3 downto 0);
    Si5324_A_DHOLD:     in  std_logic;
    Si5324_A_FOS2:      out std_logic := '0';
    Si5324_A_FOS1:      out std_logic := '0';
    Si5324_A_LOL:       out std_logic := '0';
    Si5324_B_clkin_sel: in  std_logic;
    Si5324_B_out_fsel:  in  std_logic_vector(3 downto 0);
    Si5324_B_in_fsel:   in  std_logic_vector(4 downto 0);
    Si5324_B_bw_sel:    in  std_logic_vector(3 downto 0);
    Si5324_B_DHOLD:     in  std_logic;
    Si5324_B_FOS2:      out std_logic := '0';
    Si5324_B_FOS1:      out std_logic := '0';
    Si5324_B_LOL:       out std_logic := '0';
    Si5324_C_clkin_sel: in  std_logic;
    Si5324_C_out_fsel:  in  std_logic_vector(3 downto 0);
    Si5324_C_in_fsel:   in  std_logic_vector(4 downto 0);
    Si5324_C_bw_sel:    in  std_logic_vector(3 downto 0);
    Si5324_C_DHOLD:     in  std_logic;
    Si5324_C_FOS2:      out std_logic := '0';
    Si5324_C_FOS1:      out std_logic := '0';
    Si5324_C_LOL:       out std_logic := '0');
end component;

component lcd_control3
generic (
    ROM_FILE_NAME :         string := "file_name.txt";
    MIN_FMC_FPGA_REVISION:  integer := 8;
    REQUIRED_CML_TYPE:      integer := 0;
    REQUIRED_CMH_TYPE:      integer := 0);
port (
    clk:                    in  std_logic;
    rst:                    in  std_logic;
    sw_c:                   in  std_logic;
    sw_w:                   in  std_logic;
    sw_e:                   in  std_logic;
    sw_n:                   in  std_logic;
    sw_s:                   in  std_logic;
    fpga_rev:               in  std_logic_vector(7 downto 0);
    cml_type:               in  std_logic_vector(15 downto 0);
    cml_type_valid:         in  std_logic;
    cml_type_error:         in  std_logic;
    cmh_type:               in  std_logic_vector(15 downto 0);
    cmh_type_valid:         in  std_logic;
    cmh_type_error:         in  std_logic;
    active_rx:              in  std_logic_vector(3 downto 0);
    rx1_locked:             in  std_logic;
    rx1_mode:               in  std_logic_vector(1 downto 0);
    rx1_level:              in  std_logic;
    rx1_t_format:           in  std_logic_vector(3 downto 0);
    rx1_m:                  in  std_logic;
    rx2_locked:             in  std_logic;
    rx2_mode:               in  std_logic_vector(1 downto 0);
    rx2_level:              in  std_logic;
    rx2_t_format:           in  std_logic_vector(3 downto 0);
    rx2_m:                  in  std_logic;
    rx3_locked:             in  std_logic;
    rx3_mode:               in  std_logic_vector(1 downto 0);
    rx3_level:              in  std_logic;
    rx3_t_format:           in  std_logic_vector(3 downto 0);
    rx3_m:                  in  std_logic;
    rx4_locked:             in  std_logic;
    rx4_mode:               in  std_logic_vector(1 downto 0);
    rx4_level:              in  std_logic;
    rx4_t_format:           in  std_logic_vector(3 downto 0);
    rx4_m:                  in  std_logic;
    sync_active:            in  std_logic;
    sync_enable:            in  std_logic;
    sync_v:                 in  std_logic;
    sync_err:               in  std_logic;
    sync_m:                 in  std_logic;
    sync_frame_rate:        in  std_logic_vector(2 downto 0);
    sync_video_fmt:         in  std_logic_vector(10 downto 0);
    lcd_e:                  out std_logic;
    lcd_rw:                 out std_logic;
    lcd_rs:                 out std_logic;
    lcd_d:                  out std_logic_vector(3 downto 0));
end component;

component icon
port (
    CONTROL0 :  INOUT STD_LOGIC_VECTOR(35 DOWNTO 0);
    CONTROL1 :  INOUT STD_LOGIC_VECTOR(35 DOWNTO 0);
    CONTROL2 :  INOUT STD_LOGIC_VECTOR(35 DOWNTO 0));
end component;

component vio
port (
    CONTROL :   INOUT STD_LOGIC_VECTOR(35 DOWNTO 0);
    CLK :       IN STD_LOGIC;
    ASYNC_IN :  IN STD_LOGIC_VECTOR(51 DOWNTO 0);
    SYNC_OUT :  OUT STD_LOGIC_VECTOR(7 DOWNTO 0));
end component;

component rx_ila
port (
    CONTROL :   INOUT STD_LOGIC_VECTOR(35 DOWNTO 0);
    CLK :       IN STD_LOGIC;
    TRIG0 :     IN STD_LOGIC_VECTOR(44 DOWNTO 0));
end component;

component tx_ila
port (
    CONTROL :   INOUT STD_LOGIC_VECTOR(35 DOWNTO 0);
    CLK :       IN STD_LOGIC;
    TRIG0 :     IN STD_LOGIC_VECTOR(58 DOWNTO 0));
end component;

component video_fifo
port (
    rst:        IN std_logic;
    wr_clk:     IN std_logic;
    rd_clk:     IN std_logic;
    din:        IN std_logic_vector(9 downto 0);
    wr_en:      IN std_logic;
    rd_en:      IN std_logic;
    dout:       OUT std_logic_vector(9 downto 0);
    full:       OUT std_logic;
    empty:      OUT std_logic;
    prog_full:  OUT std_logic);
end component;

begin

--------------------------------------------------------------------------------
-- Clock inputs and outputs

-- Drive tx2_usrclk to SMA connectors for monitoring purposes
USERGPIO : OBUFDS
generic map (
    IOSTANDARD  => "LVDS_25")
port map (
    I           => tx2_usrclk,
    O           => USER_SMA_GPIO_P,
    OB          => USER_SMA_GPIO_N);

-- 148.5 MHz MGT RX reference clock input
MGTCLKIN0 : IBUFDS_GTXE1
port map (
    I           => FMC_HPC_GBTCLK0_M2C_P,
    IB          => FMC_HPC_GBTCLK0_M2C_N,
    CEB         => '0',
    O           => mgtclk_rx,
    ODIV2       => open);

-- 148.X MHz MGT TX reference clock input
MGTCLKIN1 : IBUFDS_GTXE1
port map (
    I           => FMC_HPC_CLK2_M2C_MGT_C_P,
    IB          => FMC_HPC_CLK2_M2C_MGT_C_N,
    CEB         => '0',
    O           => mgtclk_tx,
    ODIV2       => open);

--
-- The 27 MHz clock from the AVB FMC card runs the main_avb_control and 
-- cm_avb_control modules.
--
HB06IBUF : IBUFDS
generic map (
    IOSTANDARD  => "LVDS_25",
    DIFF_TERM   => TRUE)
port map (
    I           => FMC_HPC_HB06_CC_P,
    IB          => FMC_HPC_HB06_CC_N,
    O           => clk_fmc_27M_in);

BUF27M : AUTOBUF
generic map (
    BUFFER_TYPE     => "BUFG")
port map (
    I           => clk_fmc_27M_in,
    O           => clk_fmc_27M);

drpclk <= clk_fmc_27M;

-- 
-- In SD mode, output the 27 MHz clock enable to Si5324 B, otherwise output rx2_usrclk.
-- Si5324 B will synthesize a 148.5 MHz or 148.35 MHz clock from this input reference
-- clock to provide a reference clock to the GTX TX.
--
Si5324_b_clkin <= rx2_ce(2) when tx2_mode_int = "01" else rx2_usrclk;

SI5324B_CLKIN2 : OBUFDS
generic map (
    IOSTANDARD  => "LVDS_25")
port map (
    I           => Si5324_b_clkin,
    O           => FMC_HPC_LA16_P,
    OB          => FMC_HPC_LA16_N);


--------------------------------------------------------------------------------
-- Triple-Rate SDI RX
--

--
-- Global clock buffer for recovered clock
--
RXBUFG : BUFG
port map (
    I           => rx2_recclk,
    O           => rx2_usrclk);

--
-- Triple rate SDI RX data path
--
SDIRX1 : triple_sdi_rx_20b
generic map (
    NUM_SD_CE               => 3,                -- 3 copies of ce_sd are used
    NUM_3G_DRDY             => 1)
port map (
    clk                     => rx2_usrclk,
    rst                     => '0',
    data_in                 => rx2_gtx_data,
    frame_en                => '1',             -- framer always enabled, no TRS filtering
    mode                    => rx2_mode,
    mode_HD                 => rx2_mode_HD,
    mode_SD                 => rx2_mode_SD,
    mode_3G                 => rx2_mode_3G,
    mode_locked             => rx2_mode_locked,
    rx_locked               => rx2_hd_locked,
    t_format                => rx2_hd_format,
    level_b_3G              => rx2_level_b,
    ce_sd                   => rx2_ce,
    nsp                     => open,
    ln_a                    => rx2_ln_a,
    a_vpid                  => rx2_a_vpid,
    a_vpid_valid            => rx2_a_vpid_valid,
    b_vpid                  => rx2_b_vpid,
    b_vpid_valid            => rx2_b_vpid_valid,
    crc_err_a               => rx2_crc_err_a,
    ds1_a                   => rx2_ds1a,
    ds2_a                   => rx2_ds2a,
    eav                     => rx2_eav,
    sav                     => rx2_sav,
    trs                     => open,
    ln_b                    => rx2_ln_b,
    dout_rdy_3G             => rx2_dout_rdy_3G,
    crc_err_b               => rx2_crc_err_b,
    ds1_b                   => rx2_ds1b,
    ds2_b                   => rx2_ds2b);

--
-- SD-SDI EDH processor
--
-- Note that the EDH processor can update the EDH packet flags. However, for
-- this demo, the video stream with updated EDH packets is not used. This is 
-- makes the demo work more like a distribution amplifier. The demo can be 
-- easily changed to make it transmit updated EDH packets by simply connecting 
-- the vid_out port of this module into the video_fifo's din port instead of 
-- rx2_ds1a.
--
rx2_edh_rst <= '1' when rx2_mode /= "01" else '0';

EDH : edh_processor
port map (
    clk                     => rx2_usrclk,
    ce                      => rx2_ce(0),
    rst                     => rx2_edh_rst,         -- keep EDH processor reset when not in SD mode
    vid_in                  => video_type(rx2_ds1a),
    reacquire               => '0',
    en_sync_switch          => '1',
    en_trs_blank            => '0',
    anc_idh_local           => '0',
    anc_ues_local           => '0',
    ap_idh_local            => '0',
    ff_idh_local            => '0',
    errcnt_flg_en           => "0000010000100000",
    clr_errcnt              => rx2_clr_errs,
    receive_mode            => '1',                  
    vid_out                 => open,
    std                     => rx2_sd_format,
    std_locked              => rx2_sd_locked,
    trs                     => open,
    field                   => open,
    v_blank                 => open,
    h_blank                 => open,
    horz_count              => open,
    vert_count              => open,
    sync_switch             => open,
    locked                  => open,
    eav_next                => open,
    sav_next                => open,
    xyz_word                => open,
    anc_next                => open,
    edh_next                => open,
    rx_ap_flags             => open,
    rx_ff_flags             => open,
    rx_anc_flags            => open,
    ap_flags                => open,
    ff_flags                => open,
    anc_flags               => open,
    packet_flags            => open,
    errcnt                  => rx2_edh_errcnt,
    edh_packet              => open);

--
-- Choose the HD/3G error, locked, and transport format signals based on the 
-- SDI mode.
--
rx2_crc_err <= (not rx2_mode_SD) and (rx2_crc_err_a or (rx2_mode_3G and rx2_level_b and rx2_crc_err_b));
rx2_locked <= rx2_sd_locked when rx2_mode_SD = '1' else rx2_hd_locked;
rx2_format <= ('0' & std_logic_vector(rx2_sd_format)) when rx2_mode_SD = '1' else rx2_hd_format;

GPIO_LED_0 <= rx2_locked;

-- Capture CRC errors for the ChipScope VIO. This FF is manually cleared by
-- a VIO module output.
--
process(rx2_usrclk)
begin
    if rising_edge(rx2_usrclk) then
        if rx2_clr_errs = '1' then
            rx2_crc_err_capture <= '0';
        elsif rx2_crc_err = '1' then
            rx2_crc_err_capture <= '1';
        end if;
    end if;
end process;
    
--------------------------------------------------------------------------------
-- RX -> TX Interface Logic
--
-- In HD and 3G modes, the data from the RX is passed to the TX data path with
-- just a set of registers for timing purposes. In SD mode, the data passes
-- through a FIFO to synchronize it to the TX clock. This is necessary because,
-- in SD mode, the RX data is not timed to a recovered clock. Instead, the RX
-- clock is really derived directly from the GTX reference clock and the cadence
-- of the RX clock enable varies slightly when the DRU needs to catch up. The
-- TX side, however, is driven by a recovered clock synthesized by the Si5324
-- and the cadence of TX clock enable never varies from 5/6/5/6 in SD mode.
-- 

--
-- Only change the tx2_mode signal after the rising edge of rx2_mode_locked.
-- This insures that the tx2_mode does change as the RX is unlocked and searching
-- for the correct SDI mode. For the most part, this logic is done in the 
-- rx2_usrclk domain because tx2_usrclk can stop, which can possibly prevent the
-- TX from being fully configured to the new SDI mode, making it impossible to
-- recover. Once the new RX is locked and the new TX mode has been established
-- in the rx2_usrclk domain, the signals are carefully moved to the tx2_usrclk
-- domain.
--
process(rx2_usrclk)
begin
    if rising_edge(rx2_usrclk) then
        rx2_mode_change_gen <= (rx2_mode_change_gen(0) & rx2_mode_locked);
    end if;
end process;

process(rx2_usrclk)
begin
    if rising_edge(rx2_usrclk) then
        if rx2_mode_change_gen = "01" then
            rx2_mode_locked_rise <= '1';
        else
            rx2_mode_locked_rise <= '0';
        end if;
    end if;
end process;

process(rx2_usrclk)
begin
    if rising_edge(rx2_usrclk) then
        if rx2_mode_locked_rise = '1' then
            tx2_mode_int <= rx2_mode;
        end if;
    end if;
end process;

process(tx2_usrclk)
begin
    if rising_edge(tx2_usrclk) then
        rx2_mode_locked_rise_txsync <= (rx2_mode_locked_rise_txsync(1 downto 0) & rx2_mode_locked_rise);
    end if;
end process;

process(tx2_usrclk)
begin
    if rising_edge(tx2_usrclk) then
        if rx2_mode_locked_rise_txsync(2) = '1' then
            tx2_mode <= tx2_mode_int;
        end if;
    end if;
end process;

--
-- Generate the tx2_out_mode bus to control the Tx output module. This bus
-- must be 00 in HD and 3G-A modes, 01 in SD mode, and 10 in 3G-B mode.
--
process(tx2_usrclk)
begin
    if rising_edge(tx2_usrclk) then
        rx2_mode_SD_syncer <= (rx2_mode_SD_syncer(0) & rx2_mode_SD);
        rx2_mode_3G_syncer <= (rx2_mode_3G_syncer(0) & rx2_mode_3G);
        rx2_level_b_syncer <= (rx2_level_b_syncer(0) & rx2_level_b);
    end if;
end process;

rx2_mode_SD_sync <= rx2_mode_SD_syncer(1);
rx2_mode_3G_sync <= rx2_mode_3G_syncer(1);
rx2_level_b_sync <= rx2_level_b_syncer(1);

process(tx2_usrclk)
begin
    if rising_edge(tx2_usrclk) then
        if rx2_mode_SD_sync = '1' then
            tx2_out_mode <= "01";
        elsif rx2_mode_3G_sync = '1' and rx2_level_b_sync = '1' then
            tx2_out_mode <= "10";
        else
            tx2_out_mode <= "00";
        end if;
    end if;
end process;

--
-- Link A data stream 2 and both data streams for link B are always directly
-- driven by the data streams from the receiver.
--
process(tx2_usrclk)
begin
    if rising_edge(tx2_usrclk) then
        if rx2_dout_rdy_3G(0) = '1' then
            tx2_ds2a <= rx2_ds2a;
            tx2_ds1b <= rx2_ds1b;
            tx2_ds2b <= rx2_ds2b;
        end if;
    end if;
end process;

--
-- Link A data stream 1 is driven by the data stream from receiver in HD and 3G
-- modes, but in SD mode, the source is the output of the video_fifo.
-- 
process(tx2_usrclk)
begin
    if rising_edge(tx2_usrclk) then
        if rx2_mode_SD_sync = '1' then
            tx2_ds1a <= fifo2_dout;
            tx2_eav <= '0';
            tx2_sav <= '0';
        elsif rx2_dout_rdy_3G(0) = '1' then
            tx2_ds1a <= rx2_ds1a;
            tx2_eav <= rx2_eav;
            tx2_sav <= rx2_sav;
        end if;
    end if;
end process;

-- 
-- FIFO for SD-SDI data path -- 16 locations deep -- asynchronous RD/WR clocks
--
-- This FIFO is used only in SD mode to allow for the phase differences between
-- the receive clock and the transmit clock. Data is written to the FIFO using
-- the rx2_usrclk, which in SD mode is locked to the RX reference clock and
-- enabled using the rx2_ce clock enable. Data is read from the FIFO using
-- tx2_usrclk, driven by TXOUTCLK in SD mode, and enabled using the tx2_ce
-- clock enabled which has a constant 5/6/5/6 cadence in SD mode.
--
-- This FIFO was generated using the Corgen FIFO generator. It uses distributed
-- RAM. The programmable full signal is set to 8.
--
FIFO1 : video_fifo
port map (
    din         => rx2_ds1a,
    rd_clk      => tx2_usrclk,
    rd_en       => tx2_ce(0),
    rst         => '0',
    wr_clk      => rx2_usrclk,
    wr_en       => rx2_ce(1),
    dout        => fifo2_dout,
    empty       => fifo2_empty,
    full        => fifo2_full,
    prog_full   => fifo2_half_full);

--
-- FIFO control logic
--
-- The prog_full signal of FIFO comes from the write clock domain. So 
-- synchronization of this signal into read clock domain is required to control 
-- the FIFO read enable. On startup, and in the event that the FIFO becomes 
-- empty, reads are disabled until the FIFO reaches the half full level.
-- 
process(tx2_usrclk)
begin
    if rising_edge(tx2_usrclk) then
        fifo2_syn <= (fifo2_syn(0) & fifo2_half_full);
    end if;
end process;
    
fifo2_half_syn <= fifo2_syn(1);

process(tx2_usrclk)
begin
    if rising_edge(tx2_usrclk) then
        if fifo2_empty = '1' then
            fifo2_rden <= '0';
        elsif fifo2_half_syn = '1' then
            fifo2_rden <= '1';
        end if;
    end if;
end process;

--------------------------------------------------------------------------------
-- TX Section
--

-- 
-- The TXUSRCLK is driven by RXRECCLK in HD and 3G modes and by TXOUTCLK in SD
-- mode. This is because RXRECCLK is a true recovered clock in HD and 3G modes,
-- but is locked to the RX GTX reference clock in SD mode.
--
-- TXOUTCLK could be used to drive tx2_usrclk in HD and 3G modes, because it is
-- frequency locked to RXRECCLK in these modes. However, the phase relationship
-- between rx2_usrclk and tx2_usrclk would not be known making it necessary to
-- synchronize the data and signals from the RX side to tx2_usrclk. By sourcing
-- tx2_usrclk from rx2_recclk in HD and 3G modes, both tx2_usrclk and rx2_usrclk
-- are sourced from the same clock signal, making them phase and frquency locked.
-- The phase difference between tx2_usrclk when driven by rx2_recclk and the
-- phase of the internal GTX transmit clocks is taken care of by the TX buffer
-- internal to the GTX when operated in this manner.
--
TXUSRCLK_BUFG : BUFGMUX
port map (
    I0      => rx2_recclk,
    I1      => tx2_outclk,
    S       => rx2_mode_SD,
    O       => tx2_usrclk);

--
-- Set the GTX TX postemphasis to optimize the TX eye. Different values are used
-- for HD and 3G. Note that these values are specific to the ML605 and FMC
-- mezzanine board. On other boards, tests must be conducted in order to determine
-- the optimimum settings.
--
tx2_postemphasis <= "10100" when tx2_mode = "10" else "01010";

--
-- Generate a TX clock enable that has a 5/6/5/6 cadence in SD mode and is 
-- always High in HD and 3G modes.
--
process(tx2_usrclk)
begin
    if rising_edge(tx2_usrclk) then
        tx2_ce_gen <= (tx2_ce_gen(9 downto 0) & tx2_ce_gen(10));
    end if;
end process;

tx2_ce_mux <= tx2_ce_gen(10) when rx2_mode_SD_sync = '1' else '1';

process(tx2_usrclk)
begin
    if rising_edge(tx2_usrclk) then
        tx2_ce <= (others => tx2_ce_mux);
    end if;
end process;

--
-- The tx2_din_rdy signal is controlled by the rx2_dout_rdy_3G signal. In all
-- modes except 3G-B, it will always be High. In 3G-B mode, it is asserted every
-- other clock cycle.
--
tx2_din_rdy <= rx2_dout_rdy_3G(0);

--
-- Triple-rate SDI TX output module.
--
-- In this pass-through configuration, EDH packets, CRC words, and line numbers
-- are not inserted. The values present in the RX data streams are simply 
-- passed through.
--
TXOUTPUT : triple_sdi_tx_output_20b
port map (
    clk             => tx2_usrclk, 
    ce              => tx2_ce,
    din_rdy         => tx2_din_rdy,
    rst             => '0',
    mode            => tx2_out_mode,
    ds1a            => tx2_ds1a,
    ds2a            => tx2_ds2a,
    ds1b            => tx2_ds1b,
    ds2b            => tx2_ds2b,
    insert_crc      => '0',
    insert_ln       => '0',
    insert_edh      => '0',
    ln_a            => "00000000000",
    ln_b            => "00000000000",
    eav             => tx2_eav,
    sav             => tx2_sav,
    txdata          => tx2_gtx_data,
    ce_align_err    => open);

--------------------------------------------------------------------------------
-- GTX Transceiver
--

--
-- When the LOL signal from CM L Si5324 B falls, hold GTXTXRESET High for 1K
-- counts of drpclk to make sure the TX reference clock is stable before
-- releasing GTXTXRESET.
--

process(drpclk)
begin
    if rising_edge(drpclk) then
        LOL_sync <= (LOL_sync(0) & cml_Si5324_B_LOL);
    end if;
end process;

process(drpclk)
begin
    if rising_edge(drpclk) then
        if LOL_sync(0) = '1' then
            gtxtxreset_dly <= (others => '0');
        elsif gtxtxreset_dly_tc = '0' then
            gtxtxreset_dly <= gtxtxreset_dly + 1;
        end if;
    end if;
end process;

gtxtxreset_dly_tc <= '1' when (gtxtxreset_dly = (gtxtxreset_dly'range => '1')) else '0';

gtxtxreset <= cml_Si5324_B_LOL or not gtxtxreset_dly_tc;

mgtrefclkrx <= ('0' & mgtclk_rx);
mgtrefclktx <= (mgtclk_tx & '0');

SDIGTX : V6SDI_WRAPPER
port map (
    ------------------------ Loopback and Powerdown Ports ----------------------
    GTX0_LOOPBACK_IN        => "000",
    ------------------- Receive Ports - RX Data Path interface -----------------
    GTX0_RXDATA_OUT         => rx2_gtx_data,
    GTX0_RXRECCLK_OUT       => rx2_recclk,
    GTX0_RXRESET_IN         => '0',
    GTX0_RXUSRCLK2_IN       => rx2_usrclk,
    GTX0_RXRATE_IN          => rx2_rate,
    ------- Receive Ports - RX Driver,OOB signalling,Coupling and Eq.,CDR ------
    GTX0_RXN_IN             => FMC_HPC_DP1_M2C_N,
    GTX0_RXP_IN             => FMC_HPC_DP1_M2C_P,
    -------- Receive Ports - RX Elastic Buffer and Phase Alignment Ports -------
    GTX0_RXBUFSTATUS_OUT    => rx2_bufstatus,
    GTX0_RXBUFRESET_IN      => rx2_bufreset,
    GTX0_RXCDRRESET_IN      => rx2_cdrreset,
    ------------------------ Receive Ports - RX PLL Ports ----------------------
    GTX0_MGTREFCLKRX_IN     => mgtrefclkrx,
    GTX0_PERFCLKRX_IN       => '0',
    GTX0_GREFCLKRX_IN       => '0',
    GTX0_NORTHREFCLKRX_IN   => "00",
    GTX0_SOUTHREFCLKRX_IN   => "00",
    GTX0_GTXRXRESET_IN      => fmc_Si5324_LOL,          -- Keep RX reset until reference clock is stable
    GTX0_RXPLLREFSELDY_IN   => "000",
    GTX0_PLLRXRESET_IN      => '0',
    GTX0_RXPLLLKDET_OUT     => rx2_pll_locked,
    GTX0_RXRATEDONE_OUT     => rx2_ratedone,
    GTX0_RXRESETDONE_OUT    => rx2_resetdone,
    ------------- Shared Ports - Dynamic Reconfiguration Port (DRP) ------------
    GTX0_DADDR_IN           => dp1_daddr,
    GTX0_DCLK_IN            => drpclk,
    GTX0_DEN_IN             => dp1_den,
    GTX0_DI_IN              => dp1_di,
    GTX0_DRDY_OUT           => dp1_drdy,
    GTX0_DRPDO_OUT          => dp1_do,
    GTX0_DWE_IN             => dp1_dwe,
    ------------------ Transmit Ports - TX Data Path interface -----------------
    GTX0_TXDATA_IN          => tx2_gtx_data,
    GTX0_TXOUTCLK_OUT       => tx2_outclk,
    GTX0_TXRESET_IN         => tx2_reset,
    GTX0_TXUSRCLK2_IN       => tx2_usrclk,
    ---------------- Transmit Ports - TX Driver and OOB signaling --------------
    GTX0_TXN_OUT            => FMC_HPC_DP1_C2M_N,
    GTX0_TXP_OUT            => FMC_HPC_DP1_C2M_P,
    GTX0_TXDIFFCTRL_IN      => "1010",
    GTX0_TXPOSTEMPHASIS_IN  => tx2_postemphasis,
    GTX0_TXPREEMPHASIS_IN   => "0000",
    ----------- Transmit Ports - TX Elastic Buffer and Phase Alignment ---------
    GTX0_TXBUFSTATUS_OUT    => tx2_bufstatus,
    ----------------------- Transmit Ports - TX PLL Ports ----------------------
    GTX0_MGTREFCLKTX_IN     => mgtrefclktx,
    GTX0_PERFCLKTX_IN       => '0',
    GTX0_GREFCLKTX_IN       => '0',
    GTX0_NORTHREFCLKTX_IN   => "00",
    GTX0_SOUTHREFCLKTX_IN   => "00",
    GTX0_GTXTXRESET_IN      => gtxtxreset,      -- Keep RX reset when reference clock is not stable
    GTX0_TXPLLREFSELDY_IN   => "001",
    GTX0_PLLTXRESET_IN      => '0',
    GTX0_TXPLLLKDET_OUT     => tx2_pll_locked,
    GTX0_TXRESETDONE_OUT    => tx2_resetdone,
    GTX0_GTXTEST_IN         => tx2_gtxtest);

--------------------------------------------------------------------------------
-- GTX control module
--
GTXCTRL : v6gtx_sdi_control
generic map (
    DRPCLK_FREQ             => 27000000)    -- The frequency of the DRP clock is 27 MHz
port map (
    drpclk                  => drpclk,
    rst                     => '0',

    txusrclk                => tx2_usrclk,
    tx_mode                 => tx2_mode_int,
    txreset_in              => '0',
    txresetdone             => tx2_resetdone,
    txbufstatus1            => tx2_bufstatus(1),
    txplllkdet              => tx2_pll_locked,
    txreset_out             => tx2_reset,
    gtxtest                 => tx2_gtxtest,
    tx_rate_change_done     => open,
    tx_slew                 => tx2_slew,

    rxusrclk                => rx2_usrclk,
    rx_mode                 => rx2_mode,
    rxresetdone             => rx2_resetdone,
    rxbufstatus2            => rx2_bufstatus(2),
    rxratedone              => rx2_ratedone,
    rxcdrreset              => rx2_cdrreset,
    rxbufreset              => rx2_bufreset,
    rxrate                  => rx2_rate,
    rx_m                    => rx2_m,

    drpdo                   => dp1_do,
    drdy                    => dp1_drdy,
    daddr                   => dp1_daddr,
    di                      => dp1_di,
    den                     => dp1_den,
    dwe                     => dp1_dwe);

--------------------------------------------------------------------------------
-- AVB FMC card interface
--

--
-- Main AVB FMC card control module
--
AVBFMC : main_avb_control
port map (
    clk                 => clk_fmc_27M,
    rst                 => '0',

-- SPI interface to AVB FMC card
    sck                 => FMC_HPC_LA00_CC_P,
    mosi                => FMC_HPC_LA00_CC_N,
    miso                => FMC_HPC_LA27_P,
    ss                  => FMC_HPC_LA14_N,

-- General status signals
    fpga_rev            => fmc_fpga_rev,
    exp_brd_prsnt       => fmc_exp_brd_prsnt,
    board_options       => open,

-- Clock XBAR control signals
--
-- For XBAR 1, each output can be driven by any of the four inputs as follows:
--      00 selects clock from Si5324
--      01 selects clock module L CLK OUT 1
--      10 selects clock module L CLK OUT 2
--      11 selects OUT 0 of XBAR 3
-- 
    xbar1_out0_sel      => "00",    -- select 148.5 MHz RX ref clock from main Si5324
    xbar1_out1_sel      => "01",    -- select 148.X MHz TX ref clock from CM L out 1
    xbar1_out2_sel      => "00",    -- not used
    xbar1_out3_sel      => "00",    -- not used

--
-- For XBAR 2, each output can be driven by any of the four inuts as follows:
--      00 selects OUT 3 of XBAR 3
--      01 selects clock module H CLK OUT 1
--      10 selects clock module H CLK OUT 2
--      11 selects clock module H CLK OUT 3
--
    xbar2_out0_sel      => "00",
    xbar2_out1_sel      => "00",
    xbar2_out2_sel      => "00",
    xbar2_out3_sel      => "00",

--
-- For XBAR 3, each output can be driven by any of the four inputs as follows:
--      00 selects FMC HA19
--      01 selects FMC LA22
--      10 selects FMC DP0 (LPC compatible MGT)
--      11 selects FMC DP1 (HPC compatible MGT)
--
    xbar3_out0_sel      => "00",    -- This output drives XBAR #1 IN3
    xbar3_out1_sel      => "10",    -- This output drives the TX1 cable driver
    xbar3_out2_sel      => "11",    -- This output drives the TX2 cable driver
    xbar3_out3_sel      => "01",    -- This output drives XBAR #2 IN0

-- Si5324 Status & Control
--
-- The Si5324_clkin_sel port controls the clock input selection for the Si5324.
-- There are three possible clock sources: 27 MHz XO, FPGA signal, and the HSYNC
-- signal from the clock separator. If the HSYNC signal is chosen, the device can be
-- put into auto frequency select mode where the controller automatically determines
-- the external HSYNC frequency and selects the proper frequency synthesis
-- settings to produce 27 MHz out of the Si5324. If Si5324_clkin_sel is anything
-- other than 01 (auto HSYNC mode), the frequency synthesis of the Si5324 is
-- controlled by the Si5324_in_fsel and Si5324_out_fsel ports as follows:
--
--      Si5324_in_fsel[4:0] select the input frequency:
--          0x00: 480i (NTSC) HSYNC
--          0x01: 480p HSYNC
--          0x02: 576i (PAL) HSYNC
--          0x03: 576p HSYNC
--          0x04: 720p 24 Hz HSYNC
--          0x05: 720p 23.98 Hz HSYNC
--          0x06: 720p 25 Hz HSYNC
--          0x07: 720p 30 Hz HSYNC
--          0x08: 720p 29.97 Hz HSYNC
--          0x09: 720p 50 Hz HSYNC
--          0x0A: 720p 60 Hz HSYNC
--          0x0B: 720p 59.94 Hz HSYNC
--          0x0C: 1080i 50 Hz HSYNC
--          0x0D: 1080i 60 Hz HSYNC
--          0x0E: 1080i 59.94 Hz HSYNC
--          0x0F: 1080p 24 Hz HSYNC
--          0x10: 1080p 23.98 Hz HSYNC
--          0x11: 1080p 25 Hz HSYNC
--          0x12: 1080p 30 Hz HSYNC
--          0x13: 1080p 29.97 Hz HSYNC
--          0x14: 1080p 50 Hz HSYNC
--          0x15: 1080p 60 Hz HSYNC
--          0x16: 1080p 59.94 Hz HSYNC
--          0x17: 27 MHz
--          0x18: 74.25 MHz
--          0x19: 74.25/1.001 MHz
--          0x1A: 148.5 MHz
--          0x1B: 148.5/1.001 MHz
--
--      Si5324_out_fsel[3:0] select the output frequency:
--          0: 27 MHz
--          1: 74.25 MHz
--          2: 74.25/1.001 MHz
--          3: 148.5 MHz
--          4: 148.5/1.001 MHz
--          5: 24.576 MHz
--          0x6: 148.5/1.0005 MHz
--
-- Note that any HSYNC frequency can only be converted to 27 MHz. Choosing any
-- output frequency except 27 MHz when the input selection is 0x00 through 0x16
-- will result in an error. Any input frequency selected by 0x17 through 0x1B
-- can be converted to any output frequency, with the exception that the
-- 74.25/1.001 and 148.5/1.001 MHz input frequencies can't be converted to 
-- 24.576 MHz.
--
-- For custom frequency synthesis, use the Si5324 register peek/poke facility
-- to modify individual registers on a custom basis.
--
    Si5324_reset        => '0',             -- 1 resets Si5324
    Si5324_clkin_sel    => "00",            -- Control input clock source selection for Si5324
                                            -- 00=27 MHz, 01=sync sep HSYNC (auto fsel mode)
                                            -- 10=FMC LA29, 11=sync sep HSYNC (manual fsel mode)
    Si5324_out_fsel     => "0011",          -- 148.5 MHz output frequency
    Si5324_in_fsel      => "10111",         -- 27 MHz input frequency
    Si5324_bw_sel       => "1010",          -- Set for 6 Hz bandwidth
    Si5324_DHOLD        => '0',
    Si5324_FOS2         => open,            -- 1=frequency offset alarm for CKIN2
    Si5324_FOS1         => open,            -- 1=frequency offset alram for CKIN1
    Si5324_LOL          => fmc_Si5324_LOL,  -- 0=PLL locked, 1=PLL unlocked

-- Si5324 register peek/poke control
    Si5324_reg_adr      => X"00",           -- Si5324 peek/poke register address (8-bit)
    Si5324_reg_wr_dat   => X"00",           -- Si5324 peek/poke register write data (8-bi)
    Si5324_reg_rd_dat   => open,            -- Si5324 peek/poke register read data (8-bit)
    Si5324_reg_wr       => '0',             -- Si5324 poke request, assert High for one clk
    Si5324_reg_rd       => '0',             -- Si5324 peek request, assert High for one clk
    Si5324_reg_rdy      => open,            -- Si5324 peek/poke cycle done when 1
    Si5324_error        => open,            -- Si5324 peek/poke error when 1 (transfer was NACKed on I2C bus)

--
-- These ports are associated with the LMH1981 sync separator.  Note that the
-- actual sync signals are available directly to the FPGA via FMC signals. The
-- sync_video_frame value is a count of the number of lines in a field or frame
-- as captured directly by the LMH1981. The sync_m and sync_frame_rate indicate
-- the frame rate of the video signal as shown below. 
--
--      sync_frame_rate     Frame Rate      sync_m
--              000         23.98 Hz            1
--              001         24 Hz               0
--              010         25 Hz               0
--              011         29.97 Hz            1
--              100         30 Hz               0
--              101         50 Hz               0
--              110         59.94 Hz            1
--              111         60 Hz               0
--
    sync_video_fmt      => open,            -- count of lines per field/frame (11-bit)
    sync_updating       => open,            -- sync_video_frame only valid when this port is 0
    sync_frame_rate     => open,            -- frame rate indicator (3-bit)
    sync_m              => open,            -- 1 = frame rate is 1000/1001
    sync_err            => open,            -- 1 = error detected frame rate

--
-- LED control ports
--

-- The eight two-color LEDs associated with the SDI RX connectors are controlled
-- by 2 bits each as follows:
--      00 = off
--      01 = green
--      10 = red
--      11 = controlled by cable EQ CD signal (green when carrier detected, else red)
--
    sdi_rx1_led         => "00",            -- controls the SDI RX1 LED
    sdi_rx2_led         => "11",            -- controls the SDI RX2 LED
    sdi_rx3_led         => "00",            -- controls the SDI RX3 LED
    sdi_rx4_led         => "00",            -- controls the SDI RX4 LED
    sdi_rx5_led         => "00",            -- controls the SDI RX5 LED
    sdi_rx6_led         => "00",            -- controls the SDI RX6 LED
    sdi_rx7_led         => "00",            -- controls the SDI RX7 LED
    sdi_rx8_led         => "00",            -- controls the SDI RX8 LED

-- All other LEDs have separate 2-bit control ports for both the red and green LEDs
-- so that the red and green sides of the LED are independently controlled like this:
--      00 = off
--      01 = on
--      10 = flash slowly
--      11 = flash quickly
--
    sdi_tx1_red_led     => "00",            -- controls the SDI TX1 red LED
    sdi_tx1_grn_led     => "00",            -- controls the SDI TX1 green LED
    sdi_tx2_red_led     => fmc_tx2_red_led, -- controls the SDI TX2 red LED
    sdi_tx2_grn_led     => fmc_tx2_grn_led, -- controls the SDI TX2 green LED
    sdi_tx3_red_led     => "00",            -- controls the SDI TX3 red LED
    sdi_tx3_grn_led     => "00",            -- controls the SDI TX3 green LED
    sdi_tx4_red_led     => "00",            -- controls the SDI TX4 red LED
    sdi_tx4_grn_led     => "00",            -- controls the SDI TX4 green LED
    sdi_tx5_red_led     => "00",            -- controls the SDI TX5 red LED
    sdi_tx5_grn_led     => "00",            -- controls the SDI TX5 green LED
    sdi_tx6_red_led     => "00",            -- controls the SDI TX6 red LED
    sdi_tx6_grn_led     => "00",            -- controls the SDI TX6 green LED
    sdi_tx7_red_led     => "00",            -- controls the SDI TX7 red LED
    sdi_tx7_grn_led     => "00",            -- controls the SDI TX7 green LED
    sdi_tx8_red_led     => "00",            -- controls the SDI TX8 red LED
    sdi_tx8_grn_led     => "00",            -- controls the SDI TX8 green LED

    aes_rx1_red_led     => "00",            -- controls the AES3 RX1 red LED
    aes_rx1_grn_led     => "00",            -- controls the AES3 RX1 green LED
    aes_rx2_red_led     => "00",            -- controls the AES3 RX2 red LED
    aes_rx2_grn_led     => "00",            -- controls the AES3 RX2 green LED
    aes_tx1_red_led     => "00",            -- controls the AES3 TX1 red LED
    aes_tx1_grn_led     => "00",            -- controls the AES3 TX1 green LED
    aes_tx2_red_led     => "00",            -- controls the AES3 TX2 red LED
    aes_tx2_grn_led     => "00",            -- controls the AES3 TX2 green LED
    madi_rx_red_led     => "00",            -- controls the MADI RX red LED
    madi_rx_grn_led     => "00",            -- controls the MADI RX green LED
    madi_tx_red_led     => "00",            -- controls the MADI TX red LED
    madi_tx_grn_led     => "00",            -- controls the MADI TX green LED
    sync_red_led        => "00",            -- controls the external sync red LED
    sync_grn_led        => "00",            -- controls the external sync green LED
    
-- SDI Cable EQ control & status
--
-- In the first two ports, there is one bit for each possible cable EQ device with
-- bit 0 for SDI RX1 and bit 7 for SDI RX8.
--
    sdi_eq_cd_n         => fmc_sdi_eq_cd_n, -- carrier detects from cable drivers, asserted low
    sdi_eq_ext_3G_reach => X"00",           -- Enable bits for extended 3G reach mode, 1=enable, 0=disable
    sdi_eq_select       => "001",           -- selects which EQ's status signals drive port below
    sdi_eq_cli          => fmc_sdi_eq_cli,  -- cable length indicator

-- SDI Cable Driver control & status
--
-- For these ports, there is one bit for each possible cable driver device with
-- bit 0 for SDI TX1 and bit 7 for SDI TX8.
--
    sdi_drv_hd_sd       => fmc_sdi_drv_hd_sd,  -- Sets slew rate of each cable driver, 1=SD, 0=HD/3G
    sdi_drv_enable      => fmc_sdi_drv_enable, -- 1 enables the driver, 0 powers driver down
    sdi_drv_fault_n     => fmc_sdi_drv_fault_n -- 1 = normal operation, 0 = fault
);

fmc_sdi_drv_hd_sd <= ("000000" & tx2_slew & '0');
fmc_sdi_drv_enable <= X"02";

-- Transmitter LEDs
fmc_tx2_grn_led <= (not fmc_sdi_drv_fault_n(1) & fmc_sdi_drv_fault_n(1)) when tx2_pll_locked = '1' else "00";
fmc_tx2_red_led <= "00" when tx2_pll_locked = '1' else "01";

--------------------------------------------------------------------------------
-- LCD Control Module
--
LCD : lcd_control3
generic map (
    ROM_FILE_NAME           => "v6sdi_pass_demo_name.txt",
    MIN_FMC_FPGA_REVISION   => 8,
    REQUIRED_CML_TYPE       => 1,   -- Si5324 clock module required in CM L
    REQUIRED_CMH_TYPE       => 0)   -- no clock module required in CM H
port map (
    clk                     => drpclk,
    rst                     => '0',
    sw_c                    => GPIO_SW_C,
    sw_w                    => GPIO_SW_W,
    sw_e                    => GPIO_SW_E,
    sw_n                    => GPIO_SW_N,
    sw_s                    => GPIO_SW_S,
    fpga_rev                => fmc_fpga_rev,
    cml_type                => cml_type,
    cml_type_valid          => cml_type_valid,
    cml_type_error          => cml_type_error,
    cmh_type                => X"0000",
    cmh_type_valid          => '0',
    cmh_type_error          => '0',
    active_rx               => "0010",  -- Only RX2 is active in this demo
    rx1_locked              => '0',
    rx1_mode                => "00",
    rx1_level               => '0',
    rx1_t_format            => "0000",
    rx1_m                   => '0',
    rx2_locked              => rx2_locked,
    rx2_mode                => rx2_mode,
    rx2_level               => rx2_level_b,
    rx2_t_format            => rx2_format,
    rx2_m                   => rx2_m,
    rx3_locked              => '0',
    rx3_mode                => "00",
    rx3_level               => '0',
    rx3_t_format            => "0000",
    rx3_m                   => '0',
    rx4_locked              => '0',
    rx4_mode                => "00",
    rx4_level               => '0',
    rx4_t_format            => "0000",
    rx4_m                   => '0',
    sync_active             => '0',
    sync_enable             => '0',
    sync_v                  => '0',
    sync_err                => '0',
    sync_m                  => '0',
    sync_frame_rate         => "000",
    sync_video_fmt          => "00000000000",
    lcd_e                   => LCD_E,
    lcd_rw                  => LCD_RW,
    lcd_rs                  => LCD_RS,
    lcd_d                   => LCD_D);

LCD_DB4 <= lcd_d(0);
LCD_DB5 <= lcd_d(1);
LCD_DB6 <= lcd_d(2);
LCD_DB7 <= lcd_d(3);

--------------------------------------------------------------------------------
-- Clock Module L interface
--
-- An Si5324 clock module must be installed in the clock module L slot on the
-- FMC mezzanine card. Si5324 B on this clock module is used to generate a low
-- jitter TX reference clock. In HD and 3G modes, this device is used as a
-- jitter cleanup PLL. In SD mode, it synthesizes a low jitter 148.5 MHz clock 
-- from the erratic 27 MHz clock enable from the SD-SDI RX.
--
CMCTRL : cm_avb_control
port map (
    clk                 => clk_fmc_27M,
    rst                 => '0',
    ga                  => '0',
    
-- SPI signals
    sck                 => FMC_HPC_LA29_P,
    mosi                => FMC_HPC_LA07_P,
    miso                => FMC_HPC_LA29_N,
    ss                  => FMC_HPC_LA07_N,

-- Module identification
    module_type         => cml_type,
    module_rev          => open,
    module_type_valid   => cml_type_valid,
    module_type_error   => cml_type_error,

-- General control
    clkin5_src_sel      => '0',         -- Clock module CLKIN 5 source
                                        -- 0 = 27 MHz, 1 = from FMC connector
-- GPIO direction signals
-- These control the direction of signals between the FPGA on the AVB FMC card
-- and the clock module. A value of 0 indicates an FPGA output to the clock
-- module. A value of 1 indicates an input to the FPGA from the clock module.
    gpio_dir_0          => X"00",
    gpio_dir_1          => X"00",
    gpio_dir_2          => X"A4",       -- 23, 21, & 18 are inputs from Si5324 Clock Module

-- General purpose output values
-- These control the of the GPIO signals when they are outputs from the FPGA
-- on the AVB FMC card to the clock module.
    gp_out_value_0      => X"00",
    gp_out_value_1      => X"00",
    gp_out_value_2      => X"00",

-- General purpose input values
-- The ports reflect the values of the GPIO signals when they are inputs to the
-- FPGA on the AVB FMC card from the clock clock module.
    gp_in_value_0       => open,
    gp_in_value_1       => open,
    gp_in_value_2       => open,
    gp_in               => open,

-- I2C bus register peek/poke control
-- These ports provide peek/poke capability to devices connected to the
-- I2C bus on the clock module. To write a value to a device register, set the
-- the slave address, register address, and data to be written and then pulse
-- i2c_reg_wr high for one cycle of the 27 MHz clock. The i2c_reg_rdy signal
-- will go low on the rising edge of the clock when i2c_reg_wr is high and
-- will stay low until the write is completed. To read a register, setup the
-- slave address and register address then pulse i2c_reg_rd high for one cycle
-- of the clock. Again, i2c_reg_rdy will go low until the read cycle is completed.
-- When i2c_reg_rdy goes high, the data read from the register will be present
-- on i2c_reg_dat_rd.
    i2c_slave_adr       => X"00",
    i2c_reg_adr         => X"00",
    i2c_reg_dat_wr      => X"00",
    i2c_reg_wr          => '0',
    i2c_reg_rd          => '0',
    i2c_reg_dat_rd      => open,
    i2c_reg_rdy         => open,
    i2c_reg_error       => open,

-- Si5324 module signals
--
-- These ports are only valid if the Si5324 clock module is installed on the
-- AVB FMC card. There are 3 identical sets of ports, one set for each of the
-- three Si5324 parts on the clock module. The out_fsel and in_fsel ports
-- set the predefined frequency synthesis options as follows:
--
--      Si5324_X_in_fsel[4:0] select the input frequency:
--          0x00: 480i (NTSC) HSYNC
--          0x01: 480p HSYNC
--          0x02: 576i (PAL) HSYNC
--          0x03: 576p HSYNC
--          0x04: 720p 24 Hz HSYNC
--          0x05: 720p 23.98 Hz HSYNC
--          0x06: 720p 25 Hz HSYNC
--          0x07: 720p 30 Hz HSYNC
--          0x08: 720p 29.97 Hz HSYNC
--          0x09: 720p 50 Hz HSYNC
--          0x0A: 720p 60 Hz HSYNC
--          0x0B: 720p 59.94 Hz HSYNC
--          0x0C: 1080i 50 Hz HSYNC
--          0x0D: 1080i 60 Hz HSYNC
--          0x0E: 1080i 59.94 Hz HSYNC
--          0x0F: 1080p 24 Hz HSYNC
--          0x10: 1080p 23.98 Hz HSYNC
--          0x11: 1080p 25 Hz HSYNC
--          0x12: 1080p 30 Hz HSYNC
--          0x13: 1080p 29.97 Hz HSYNC
--          0x14: 1080p 50 Hz HSYNC
--          0x15: 1080p 60 Hz HSYNC
--          0x16: 1080p 59.94 Hz HSYNC
--          0x17: 27 MHz
--          0x18: 74.25 MHz
--          0x19: 74.25/1.001 MHz
--          0x1A: 148.5 MHz
--          0x1B: 148.5/1.001 MHz
--
--      Si5324_X_out_fsel[3:0] select the output frequency:
--          0x0: 27 MHz
--          0x1: 74.25 MHz
--          0x2: 74.25/1.001 MHz
--          0x3: 148.5 MHz
--          0x4: 148.5/1.001 MHz
--          0x5: 24.576 MHz
--          0x6: 148.5/1.0005 MHz
--
-- Note that any HSYNC frequency can only be converted to 27 MHz. Choosing any
-- output frequency except 27 MHz when the input selection is 0x00 through 0x16
-- will result in an error. Any input frequency selected by 0x17 through 0x1B
-- can be converted to any output frequency, with the exception that the
-- 74.25/1.001 and 148.5/1.001 MHz input frequencies can't be converted to 
-- 24.576 MHz.
--
-- Only Si5324_A is currently used in this demo. It generates a 148.35 MHz
-- reference clock for the SDI transmitters.
--
    Si5324_A_clkin_sel  => '0',
    Si5324_A_out_fsel   => "0000",
    Si5324_A_in_fsel    => "10111", 
    Si5324_A_bw_sel     => "1010",
    Si5324_A_DHOLD      => '0',
    Si5324_A_FOS2       => open,
    Si5324_A_FOS1       => open,
    Si5324_A_LOL        => open,

    Si5324_B_clkin_sel  => '1',         -- select FMC LA16 as source
    Si5324_B_out_fsel   => Si5324_B_out_fsel,
    Si5324_B_in_fsel    => Si5324_B_in_fsel,
    Si5324_B_bw_sel     => Si5324_B_bw_sel,
    Si5324_B_DHOLD      => '0',
    Si5324_B_FOS2       => open,
    Si5324_B_FOS1       => open,
    Si5324_B_LOL        => cml_Si5324_B_LOL,

    Si5324_C_clkin_sel  => '0',         -- Not used
    Si5324_C_out_fsel   => "0000",      
    Si5324_C_in_fsel    => "10111",       
    Si5324_C_bw_sel     => "1010",
    Si5324_C_DHOLD      => '0',
    Si5324_C_FOS2       => open,
    Si5324_C_FOS1       => open,
    Si5324_C_LOL        => open);

-- Asserted low resets to the 3 Si5324 parts on CM L. Always keep these signals
-- High.
FMC_HPC_LA13_P <= '1';
FMC_HPC_LA33_P <= '1';
FMC_HPC_LA26_P <= '1';

-- 
-- Synchronize mode & rate signal from rx2_usrclk to clk_fmc_27M for use in
-- setting the frequency synthesis mode of Si5324 B on clock module L.
--
process(clk_fmc_27M)
begin
    if rising_edge(clk_fmc_27M) then
        tx2_mode_sync1 <= tx2_mode_int;
        tx2_mode_sync2 <= tx2_mode_sync1;
        rx2_m_sync <= (rx2_m_sync(0) & rx2_m);
    end if;
end process;

--
-- Si5324 B must run at 148.5 MHz in SD-SDI mode. In HD and 3G modes, rx2_m
-- determines the frequency. 
--
process(clk_fmc_27M)
begin
    if rising_edge(clk_fmc_27M) then
        if tx2_mode_sync2 = "01" then
            Si5324_B_out_fsel <= "0011";    -- Generate 148.5 MHz in SD mode
        elsif rx2_m_sync(1) = '1' then
            Si5324_B_out_fsel <= "0100";    -- Generate 148.35 MHz
        else
            Si5324_B_out_fsel <= "0011";    -- Generate 148.5 MHz
        end if;
    end if;
end process;

--
-- The Si5324 B input frequency is 27 MHz in SD mode, 74.25 or 74.1758 MHz in 
-- HD mode (selected by rx2_m), and 148.5 MHz or 148.35 MHz in 3G mode (again
-- selected by rx2_m).
--
process(clk_fmc_27M)
begin
    if rising_edge(clk_fmc_27M) then
        if tx2_mode_sync2 = "01" then
            Si5324_B_in_fsel <= "10111";
        elsif tx2_mode_sync2 = "00" then
            if rx2_m_sync(1) = '1' then
                Si5324_B_in_fsel <= "11001";
            else
                Si5324_B_in_fsel <= "11000";
            end if; 
        else
            if rx2_m_sync(1) = '1' then
                Si5324_B_in_fsel <= "11011";
            else
                Si5324_B_in_fsel <= "11010";
            end if;
        end if;
    end if;
end process;

--
-- The Si5324 B bandwidth is set to about 6 Hz in SD-SDI mode to provide very
-- clean filtering of the erratic 27 MHz RX clock enable as it is converted to
-- a 148.5 MHz reference clock for the TX. In HD and 3G modes, the bandwdith
-- is set to about 500 Hz in order to provide for quick lock times.
--
process(clk_fmc_27M)
begin
    if rising_edge(clk_fmc_27M) then
        if tx2_mode_sync2 = "01" then
            Si5324_B_bw_sel <= "1010";
        else
            Si5324_B_bw_sel <= "0100";
        end if;
    end if;
end process;

--------------------------------------------------------------------------------
-- ChipScope modules

i_icon : icon
port map (
    CONTROL0    => control0,
    CONTROL1    => control1,
    CONTROL2    => control2);

i_vio : vio
port map (
    CONTROL     => control0,
    CLK         => tx2_usrclk,
    ASYNC_IN    => async_in,
    SYNC_OUT    => sync_out);

async_in <= ("00000" & fmc_sdi_eq_cli & Si5324_B_out_fsel(2 downto 0) & tx2_pll_locked & rx2_m & 
                   cml_Si5324_B_LOL & fmc_Si5324_LOL & rx2_edh_errcnt & rx2_crc_err_capture & 
                   rx2_level_b & rx2_format & rx2_locked & rx2_mode & rx2_mode_locked & rx2_pll_locked);

rx2_clr_errs <= sync_out(0);


i_rx_ila : rx_ila
port map (
    CONTROL     => control1,
    CLK         => rx2_usrclk,
    TRIG0       => rx_trig0);

rx_trig0 <= (rx2_crc_err_a & rx2_dout_rdy_3G(0) & rx2_ds2b & rx2_ds1b & rx2_ds2a & rx2_ds1a & 
             rx2_sav & rx2_eav & rx2_ce(1));


i_tx_ila : tx_ila
port map (
    CONTROL     => control2,
    CLK         => tx2_usrclk,
    TRIG0       => tx_trig0);

tx_trig0 <= (tx2_bufstatus & tx2_gtx_data & tx2_sav & tx2_eav & tx2_ds2a & tx2_ds1a & tx2_ce(0) & 
             fifo2_dout & fifo2_rden & fifo2_half_full & fifo2_empty & fifo2_full);

end xilinx;

    
